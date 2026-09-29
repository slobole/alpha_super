# NDX momentum pod: why ROC12, why 10 stocks, why these filters? (2026-09-26)

Research only. No live pod, release, scheduler, broker code or WIRED strategy file was changed.
Frozen plan: [NDX_PARAM_ROBUSTNESS_PREREG_20260926.md](NDX_PARAM_ROBUSTNESS_PREREG_20260926.md), frozen before any grid
cell was computed (SHA-256 `0cc15db5...`, `results/research/ndx_param_robustness_20260926/prereg_freeze.json`; not a git
commit, see amendment A2). Post-result amendments A1-A6, none of which changes a verdict:
[NDX_PARAM_ROBUSTNESS_PREREG_20260926_AMENDMENTS.md](NDX_PARAM_ROBUSTNESS_PREREG_20260926_AMENDMENTS.md).
Code: `scripts/research/ndx_param_robustness_core.py`, `run_ndx_param_robustness_study.py`,
`check_ndx_param_robustness_invariance.py`, `run_ndx_param_robustness_engine_parity.py`,
`analyze_ndx_param_robustness_study.py`; tests `tests/test_ndx_param_robustness_study.py`.
Results, tables and charts: `results/research/ndx_param_robustness_20260926/` (`results.json`, `charts/`).

## תקציר בעברית

**הטענה שנבדקה.** האם ההגדרות של פוד ה-NDX (תשואת 12 חודשים, 10 מניות, מסנן ממוצע 100 ימים, שער SPY, סקיילר VXN) נבחרו טוב, או שיש אזור יציב וטוב יותר?

**איך בדקנו.**
- בנינו חמש רשתות קטנות, כל אחת משנה שאלה אחת בלבד, והכול הוקפא מראש.
- כל הגרסאות חופשיות מיחידות דולר, ולכן לא תלויות בתאריך ההורדה של הנתונים (אומת).
- שפטנו לפי "רמה" (חציון של תא ושכניו), לא לפי התא הטוב ביותר.
- הכלל שהוקפא: שינוי מחליף את הגרסה החיה (L) רק אם השכונה שלו מנצחת אותה בתוך G3 בכל שלוש התקופות, בלי ירידה גרועה ביותר מ-2 נקודות אחוז.

**התוצאה.**
- אף שינוי לא עבר את הכלל. ממליצים להשאיר את L.
- כל המועמדים ניצחו את L בשנים 2008-2011 וב-2012-2021, וכולם הפסידו לה ב-2022-2026.
- ההפסד הזה הוא בעיקר שנה אחת: ב-2025 הגרסה החיה עשתה 36.5% מול 12.5% לתאומה חסרת היחידות.
- בלי 2025 המדד של L בתקופה האחרונה יורד מ-1.29 ל-1.02, כמעט כמו התאומה (0.97).
- בנוסף, כל המועמדים הביאו ירידה מקסימלית גרועה ב-2.5 עד 2.7 נקודות אחוז בתיק G3.

**למה 12 חודשים?**
- תקופות ארוכות (9-12 חודשים ושילובים שלהן) עדיפות על 3-6 חודשים ב-NDX.
- ה-12 נמצא בתוך רצועה רחבה וטובה, לא על שיא חד.
- חלוקה בתנודתיות מקטינה בעיקר את הירידות, ופחות מעלה תשואה.
- הדילוג על החודש האחרון (12-1) גרוע יותר כאן.
- אזהרה: ב-S&P 500 וב-Russell 1000 הצורה אחרת לגמרי, ולכן "12" אינו אמת כללית.

**למה 10 מניות?**
- ב-NDX עשר מניות הן בדיוק שיא המשטח (שארפ 0.83).
- פחות מניות מעלות תשואה וירידות, יותר מניות מורידות את שתיהן.
- שקילה לפי תנודתיות הפוכה לא עוזרת.
- במדדים הגדולים יותר עדיף מספר גדול יותר (20-30), כלומר 10 מתאים לגודל המאגר של NDX.

**למה המסננים?**
- ממוצע 100 ימים אינו מיוחד: 200 ימים או בלי מסנן כלל נותנים לבד תוצאה דומה או טובה יותר.
- שער ה-SPY מקטין מאוד את הירידות כשהפוד עומד לבד (מכ-35%-41% לכ-27%-29%).
- בתוך G3 השער דווקא מזיק, כי רגל ה-TAA כבר עושה את התזמון.
- ממצא שלא נבדק מראש: בלי מסנן ובלי שער, G3 מגיע לשארפ 1.40 ומנצח את L בכל שלוש התקופות.
- אבל הירידה המקסימלית שלו גרועה ב-2.8 נקודות, ולבד הוא ירד 28% ב-2008. זו השערה למחקר חדש, לא מסקנה.

**מאגר (באפר) ויום האיזון.**
- באפר לא משפר, רק מוריד מעט מחזור.
- יום האיזון משנה הרבה: על פני 21 ימים שונים השארפ של הפוד לבד נע בין 0.66 ל-0.79 בגרסה החיה.
- סוף החודש יצא כמעט הטוב ביותר, כלומר במידה מסוימת היה לנו מזל.

**סקיילר ה-VXN.**
- הרצפה כמעט לא משנה, כי היא נכנסת לפעולה רק כשה-VXN גבוה מאוד.
- היעד הוא בעצם כפתור סיכון: יעד נמוך נותן פחות תשואה ופחות ירידה, והשארפ לבד כמעט לא זז.
- בתוך G3 הסקיילר מקטין את הירידה (מ-21.1% בלעדיו ל-16.7%).

**עלויות ונזילות.**
- תוספת של 5 נקודות בסיס לכל צד לא שינתה אף הכרעה.
- מדד נזילות גס: הגרסה החיה נוטה למניות זולות וקטנות, ולכן הקיבולת שלה בשנים האחרונות נמוכה פי 1.7 עד 2.7 מהתאומה.
- לגודל החשבון הנוכחי זה לא משנה.

**בשורה התחתונה.**
- משאירים את הגרסה החיה, כי היא ברירת המחדל ואף שינוי לא עבר את הכלל. זה לא אומר שהיא הוכחה כטובה יותר.
- היתרון שלה על התאומות נשען על שנה אחת, והפער בין הגרסאות אינו מובהק סטטיסטית בכיוון כלשהו.
- כשהפוד עומד לבד, התאומה דווקא טובה יותר (0.83 מול 0.76), וכך גם 69% מכל התאים.
- ההגדרות הנוכחיות סבירות: 10 מניות ו-12 חודשים יושבים על רמה, לא על שיא מקרי.
- לפי הכלל שהוקפא, קו הצל הוא התאומה חסרת היחידות עם מסנן 200 ימים.
- כדאי לפתוח מחקר חדש ומוקפא על "בלי שער בתוך G3" ועל פיזור יום האיזון.

## Bottom line (English)

- **Keep L - as the default, not as a proven winner.** No stage candidate passed the frozen rule. Every one beat L inside G3 in 2008-11 and 2012-21 and lost in
  2022-26 (by 0.19-0.26 Sharpe), and each had a 2.5-2.7 pp worse G3 max drawdown on 2012-26.
- **L's lead is one year.** In 2025 L made +36.5% against +12.5% for its scale-free twin A0. Excluding 2025, the G3
  2022-26 Sharpe is 1.02 for L and 0.97 for A0. A0 beat L in 16 of 27 calendar years.
- **The live settings sit on plateaus, not spikes.** N = 10 is the top of the NDX N surface; ROC12/NATR20 is the plateau
  centre of the score grid; the VXN floor is irrelevant and the target is a risk dial.
- **They are not universal truths.** The score and N surfaces have different shapes on S&P 500 and Russell 1000
  (rank correlation of the score grid with NDX: -0.16 and -0.04).
- **Nothing is significant.** White's Reality Check over 151 configurations: p = 0.61 in G3 and 0.61 standalone.
  Standalone, A0 beats L (0.83 vs 0.76) and so do 69% of all cells; every R1 failure comes from the 4.5-year 2022-26
  block, a gap already known when the plan was frozen.
- **Shadow line (by the frozen rule):** A0 with an SMA200 stock filter (ROC12/NATR20, SMA200, SPY gate, N 10, VXN 22/0.25).
- **Worth a new frozen study:** the NDX leg without its own regime gate inside G3, and spreading the rebalance over
  several days (tranching). Both are post-hoc observations here.

## 1. What was tested

Anchor **A0** = the live design with L's dollar ATR replaced by its scale-free twin (ROC12 / NATR20, = S20 of the stage-3
redesign). Each stage varies one question with everything else at A0 (no chaining):

| Stage | Grid | Cells |
|---|---|---|
| S1 score | numerator ROC3, ROC6, B3612, ROC9, B612, ROC12, ROC12-1 x denominator none, NATR63, NATR20 | 21 |
| S2 size and weights | N 5, 8, 10, 15, 20, 30 x equal weight / inverse sigma63 | 12 |
| S3 filters | stock filter none, SMA50, SMA100, SMA200 x regime none, SPY>SMA200, QQQ>SMA200 | 12 |
| S4 buffer and timing | exit buffer 0, 2, 5, 10 x rebalance offset -10..+10 sessions | 84 |
| S5 VXN | target 18-26 x floor 0-0.5 (+ no-scaler reference) | 25 + 1 |

151 distinct configurations (A0 repeats), plus L at all 21 offsets (timing-luck diagnostic) and B (contrast).
Same grids on PIT S&P 500 and Russell 1000 (standalone). Engine costs, +5 bps/side stress, commission on real share
counts (secondary).

## 2. Checks before reading any result

- **Engine replica parity.** The fast replica reproduces the real engine runs of L, B, S20 and R from the earlier
  studies to the cent (max daily difference 9.5e-10 = CSV rounding; identical top 10 on all 233 invested months; repo
  schedule identical). Five new real-engine runs executing the replica's targets (N20 inverse-vol, buffer 5 at k+7,
  buffer 2 at k-8, SMA200 + QQQ gate, VXN 18/0.5) match to 7e-16 per day. Runtime: 0.2 s per run instead of ~4 min.
- **Download-date invariance.** V1: random per-stock price constants (0.01x-100x) leave all 151 configurations' lists
  and weights identical on every one of 48,320 decision-dates; B changes on 100% of dates. V2: recomputing every
  feature from Norgate unadjusted bars restated to each decision day reproduces all stage-1 lists and L exactly
  (B differs on 228 of 233 dates). Largest feature difference 4.5e-6 (NATR20; Norgate factor rounding across O/H/L/C).
- V2 extended after the review (amendment A6): every month-end cell of every stage (73 configurations, including
  inverse-vol weights, SMA50/200, QQQ gate, VXN settings, buffers) is identical from the decision-day view. The NaN
  mismatches (sigma63 24 name-dates, ROC12-1 151) involve no name that was a PIT member with a close that day.
- Engine parity for new cells checks the accounting; the selection logic of buffers, offsets, IV and the QQQ gate is
  checked by the unit tests and V1/V2, not by an independent engine implementation.
- Unit tests: 4 pass (scale invariance, buffer, offset/ROC anchors, sizing/fill/commission/dividend arithmetic).

## 3. Reference numbers

| Leg | Standalone 2000-26 CAGR / Sharpe / MaxDD | Sharpe P1 / P2 / P3 | G3 Sharpe G-P1 / G-P2 / G-P3 | G3 2012-26 CAGR / Sharpe / MaxDD |
|---|---|---|---|---|
| **L (live)** | 11.9% / 0.76 / -29.7% | 0.52 / 1.02 / 0.83 | 0.70 / 1.30 / 1.29 | 20.0% / 1.30 / -14.1% |
| A0 (scale-free twin) | 14.9% / 0.83 / -27.1% | 0.72 / 1.13 / 0.58 | 0.87 / 1.33 / 1.06 | 21.1% / 1.24 / -16.7% |
| B (biased, not implementable) | 17.6% / 1.03 / -23.2% | 0.83 / 1.36 / 0.86 | 0.78 / 1.51 / 1.29 | 23.5% / 1.44 / -14.7% |
| TAA alone | – | – | 0.77 / 1.29 / 1.42 | 23.8% / 1.32 / -17.7% |

P1 2000-11, P2 2012-21, P3 2022-26 (to 2026-07-24). G-P1 2008-11 uses the synthetic TAA proxy. Sharpe has no risk-free rate.

## 4. Frozen rule: stage candidates versus L

Candidate = cell with the best neighbourhood-median standalone 2000-26 Sharpe. Margins are neighbourhood medians minus L.

| Stage | Plateau centre | A0 rank (standalone) | G3 margin G-P1 / G-P2 / G-P3 | G3 DD gap 2012-26 / 2008-26 | R3 (+5 bps) | R4 SP500 / R1000 | Verdict |
|---|---|---|---|---|---|---|---|
| S1 score | A0 itself | 3 / 21 | +0.16 / +0.03 / **-0.26** | **-2.5 pp** / -1.2 pp | fail | trivial | fail |
| S2 N, weights | A0 itself (N 10 EW) | 1 / 12 | +0.16 / +0.03 / **-0.23** | **-2.6** / -1.4 | fail | trivial | fail |
| S3 filters | SMA200 + SPY | 5 / 12 | +0.19 / +0.04 / **-0.19** | **-2.7** / -1.5 | fail | fail / pass | fail |
| S4 buffer | buffer 2 | 5 / 84 | +0.15 / +0.03 / **-0.23** | **-2.6** / -1.4 | fail | pass / pass | fail |
| S5 VXN | 22 / 0.5 | 17 / 25 | +0.18 / +0.03 / **-0.23** | **-2.6** / -1.4 | fail | fail / fail | fail |

- Commission on real share counts changes every margin by less than 0.003. The +5 bps stress by less than 0.004.
- Share of all cells beating L in G-P3: S1 0%, S2 0%, S3 42%, S4 33%, S5 0%. In all three G3 blocks: 0-8%.
- **Shadow line by the frozen rule:** the candidate with the largest worst-block margin, S3 = A0 with SMA200.
- **Walk-forward diagnostic** (candidate chosen on 2000-11 only): S1 picks B612/NATR20, S2 N 15, S3 SMA200, S4 buffer 0,
  S5 26/0.375. Out of sample their gain over A0 is at most +0.05 Sharpe in 2012-21 or 2022-26, and inside G3 all trail L
  in 2022-26 (0.99-1.10 against 1.29).

## 5. What each grid says

**S1 - what to rank on.** Standalone 2000-26 Sharpe:

| | none | NATR63 | NATR20 |
|---|---|---|---|
| ROC3 | 0.73 | 0.76 | 0.66 |
| ROC6 | 0.73 | 0.75 | 0.71 |
| B3612 | 0.78 | 0.81 | 0.82 |
| ROC9 | 0.75 | 0.81 | 0.78 |
| B612 | 0.81 | 0.85 | 0.89 |
| ROC12 | 0.79 | 0.83 | **0.83** (A0) |
| ROC12-1 | 0.70 | 0.76 | 0.73 |

- Longer horizons win on NDX; ROC12 sits in a band of 0.78-0.89 from B3612 to ROC12.
- Volatility scaling mainly cuts drawdown: ROC12 alone -33.4%, with NATR -27.1%. NATR20 and NATR63 are equivalent.
- Skipping the last month (12-1) hurts on this universe.
- Other universes: on S&P 500 ROC9/NATR20 is best (0.79) and ROC12 among the worst (0.54-0.58). Surface rank
  correlation with NDX: -0.16 (S&P 500), -0.04 (Russell 1000). The lookback choice does not transfer.

**S2 - how many, how weighted.** Standalone Sharpe EW: N5 0.76, N8 0.79, **N10 0.83**, N15 0.81, N20 0.77, N30 0.75.
- CAGR falls with N (15.4% -> 10.7%) and drawdown shrinks (-35.7% -> -23.6%).
- Inverse-vol weights lower both return and drawdown; Sharpe is no better than equal weight (0.64-0.81), except a
  marginal +0.02 at N 30.
- On S&P 500 and Russell 1000 Sharpe rises with N up to 20-30: "10" fits the small NDX pool, not momentum in general.

**S3 - filters.** Standalone Sharpe / G3 2012-26 Sharpe:

| regime \ stock filter | none | SMA50 | SMA100 | SMA200 |
|---|---|---|---|---|
| none | 0.88 / **1.40** | 0.76 / 1.30 | 0.82 / 1.33 | 0.83 / 1.39 |
| SPY > SMA200 | 0.88 / 1.28 | 0.83 / 1.24 | **0.83 / 1.24** (A0) | 0.88 / 1.28 |
| QQQ > SMA200 | 0.80 / 1.27 | 0.78 / 1.25 | 0.77 / 1.23 | 0.80 / 1.28 |

- SMA100 is not special; SMA200 or no stock filter are at least as good.
- The SPY gate is what protects the standalone pod (max DD about -27% to -29% with it, -35% to -41% without).
- Inside G3 the gate costs Sharpe in every column, because the TAA leg already times the market.
- A QQQ gate is weaker than SPY standalone and similar inside G3 (better in 2022-26, worse before).

**S4 - buffer and timing luck.** Median standalone Sharpe over the 21 rebalance days: buffer 0 0.78, 2 0.77, 5 0.78,
10 0.75. A buffer of 2 cuts turnover from 8.3x to 7.4x a year and changes nothing else.

| Over 21 rebalance days | Standalone Sharpe min / median / max | Month-end percentile | G3 2012-26 Sharpe min / median / max |
|---|---|---|---|
| L | 0.66 / 0.70 / 0.79 | 95th | 1.24 / 1.30 / 1.34 |
| A0 | 0.68 / 0.78 / 0.89 | 90th | 1.19 / 1.25 / 1.35 |
| L, common start 2000-04-01 (A3) | 0.65 / 0.69 / 0.77 | 86th | – |
| A0, common start 2000-04-01 (A3) | 0.66 / 0.77 / 0.88 | 90th | – |

k > 0 schedules first trade in mid-February 2000, so the first two rows mix start dates; the common-start rows fix that.

- Month-end is a lucky day for both standalone; a typical day is about 0.05-0.06 Sharpe lower.
- L beats A0 in G-P3 on all 21 days and loses to A0 in G-P1 on all 21 days: the 2022-26 gap is not timing luck.
  Month-end is in fact one of L's weaker days in G-P3 (1.29, 19th percentile; median over days 1.40).

**S5 - VXN.** The floor binds only when VXN > target / floor (above 88 for 22/0.25), so it is inert.
The target is a risk dial:
- target 18: CAGR 13.3%, max DD -23.8%, G3 Sharpe 1.26, G3 DD -15.4%;
- target 26: CAGR 15.7%, max DD -29.4%, G3 Sharpe 1.22, G3 DD -18.0%;
- standalone Sharpe is flat (0.83-0.84);
- no scaler: standalone 0.82, G3 1.20 with G3 DD -21.1%.
Its value is drawdown control inside G3.

## 6. Multiplicity

- **Reality Check, G3 2012-26 Sharpe vs L** (151 configurations, stationary bootstrap, 2,000 draws): the best cell
  (no filter, no regime) is +0.11; p = 0.61. 13% of cells are above L.
- **Reality Check, standalone 2000-26 vs L:** best +0.13 (A0 at offset +7); p = 0.61. 69% of cells are above L.
  From the common start 2000-04-01 (A3): best +0.14, p = 0.59, 72% above L.
- **Paired, A0 vs L in G3:** -0.06, 90% interval [-0.20, +0.08]. SMA200 candidate: -0.01 [-0.16, +0.13].
- **Deflated Sharpe** (159 trials, cross-trial sd 0.05): about 1.00 for every candidate and for L. It only tests
  Sharpe > 0, and the cross-trial spread is small because the 84 stage-4 cells are near-duplicates, so it is not
  informative here. 159 is a lower bound: the older sweeps on B and whatever originally chose L's parameters are not in it.

## 7. Other universes (standalone 2000-26, identical rules)

| Universe | A0 Sharpe | L Sharpe | B Sharpe | Rank corr. with NDX, all cells |
|---|---|---|---|---|
| S&P 500 | 0.58 | 0.69 | 0.91 | 0.25 |
| Russell 1000 | 0.74 | 0.55 | 0.91 | 0.44 |

- L's dollar-ATR tilt helps on S&P 500 and hurts on Russell 1000: it is not a reliable feature.
- By stage, the correlation is moderate-to-high only for the VXN (0.61) and buffer (0.53) grids on S&P 500 and the
  filter grid (0.82) on Russell 1000; the score grid is negative on both.

## 8. Turnover and capacity proxy

| | Turnover x/yr | Names replaced per rebalance | Pod AUM at which the 95th-pct order = 1% of ADV20 (2000-26 / 2021-26) | Pod AUM at which the largest order = 1% of ADV20 (2000-26 / 2021-26) |
|---|---|---|---|---|
| L | 7.9 | 38% | $5.7M / $14.4M | $0.8M / $3.6M |
| A0 | 8.3 | 38% | $7.0M / $24.5M | $1.2M / $9.6M |
| SMA200 shadow | 7.9 | 35% | $7.1M / $23.4M | $1.1M / $9.6M |
| buffer 2 | 7.4 | 33% | $7.2M / $25.2M | $1.2M / $9.3M |

A proxy only (order notional / 20-day median dollar volume), not the CapacityAnalysis v2 Recommended Max. Fills are
market-on-open, and the opening auction is a small share of daily volume, so these figures overstate open-auction
capacity; the comparison between legs still holds. L's cheap-stock tilt roughly halves the recent capacity.

## 9. Post-hoc observations (not pre-registered; change no verdict)

- **2025.** A0 minus L by year: 2020 +21.5 pp, 2021 +17.3, 2024 +9.1, 2025 -24.0. Ex-2025 G3 2022-26 Sharpe:
  L 1.02, A0 0.97, SMA200 1.06, no-gate 1.29.
- **No stock filter, no regime gate, inside G3.** G3 Sharpe 0.82 / 1.42 / 1.38 by block (L 0.70 / 1.30 / 1.29), 2012-26
  1.40 at 25.5%/yr, but G3 max DD -16.9% (L -14.1%), so it would fail the drawdown condition. Standalone it lost 28% in
  2008 and has a -35% max DD: it only makes sense as a G3 leg, never alone. It is the best of 151 cells (Reality Check
  p = 0.61), so it needs its own frozen test.
- **Rebalance tranching.** Averaging the 21 rebalance-day versions (an approximation of splitting the pod into daily
  tranches): L G3 2012-26 Sharpe 1.32 with -16.1% max DD; A0 1.28 with -18.9%. It removes the month-end luck without
  changing the conclusion.

## 10. Limits

- One history; a 14-year G3 Sharpe has a standard error of about 0.27. Block differences of a few hundredths are noise.
- G-P1 rests on the owner's synthetic TAA proxy.
- The replica equals the engine for this strategy family only (monthly target weights, market orders at the open).
- L is what a day-t snapshot computes; live fills were not inspected (they are on the VPS).
- The capacity numbers are a proxy.
- The TAA leg is taken as stored; it was not re-audited here.
- Two plateau centres are grid edges (SMA200 in stage 3, floor 0.5 in stage 5), so those grids may not bracket the best
  region; the stage-5 edge over A0 (0.834 vs 0.833) is noise because the floor almost never binds.
- Shared simplifications that hit all cells alike: G3 is a frictionless daily 50/50; cash earns no interest (lowers CAGR
  of low-exposure cells, not Sharpe); commission is on split-adjusted share counts (the real-share version is reported).

## 11. Review

Independent quant-pitfalls review (read-only agent): no material findings. Lookahead, PIT membership, the replica's
engine semantics, the rule code, the G3 splice and the multiplicity code were checked. Minor findings were fixed or
disclosed as amendments A1-A6 (IV budget reading, freeze evidence, offset start dates, ADV window, literal A0 handling,
full V2 coverage), plus the capacity and DSR caveats above.
