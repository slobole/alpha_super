# New strategy search: merger arbitrage, return seasonality, and what actually fills the book's slot (2026-09-27)

Research only. Nothing live, released, scheduled or WIRED was changed, and nothing was committed.

**Who asked for what.** The owner asked for "another strategy to trade", with any idea allowed and no bias toward the
earlier trend / breakout ideas.

**Who did what.**
- The lead (Opus 5.5) chose the families, wrote and froze both plans, verified key numbers and ran the descriptive
  slot table in section 5.
- A research agent (Fable 5.1) built and ran both studies on the house engine replica.
- An independent quant-pitfalls reviewer (Opus 5.5) checked both studies; see section 4.

**Frozen plans:**
- [NEW_POD_SEARCH_PREREG_20260927.md](NEW_POD_SEARCH_PREREG_20260927.md) (SHA-256 `69fafb7b...`, 19:17): merger
  arbitrage v1 and return seasonality. Post-result amendments are in
  [NEW_POD_SEARCH_PREREG_20260927_AMENDMENTS.md](NEW_POD_SEARCH_PREREG_20260927_AMENDMENTS.md);
- [MERGER_ARB_V2_PREREG_20260927.md](MERGER_ARB_V2_PREREG_20260927.md) (SHA-256 `d5540836...`, 20:04): merger
  arbitrage v2, a disclosed follow-up.

**Earlier the same day:** [TREND_BREAKOUT_REPORT_20260927.md](TREND_BREAKOUT_REPORT_20260927.md).

**Code:** `scripts/research/new_pod_search_20260927/`, `scripts/research/merger_arb_v2_20260927/` and their tests.

**Results:** `results/research/new_pod_search_20260927/` and `results/research/merger_arb_v2_20260927/`
(`results.json`, `report_extract.json`, `table_rule.csv`, charts).

## תקציר בעברית

**מה ביקשת.** אסטרטגיה נוספת למסחר, מכל כיוון, בלי הטיה לרעיונות הקודמים.

**הממצא הראשון: הרף הנכון.**
- החלפת רבע מהתיק, מחצי פוד ה-NDX, באג"ח ממשלתיות קצרות (T-bills) משפרת את G3 בכל שלוש התקופות.
  - השארפ עולה מ-1.29 ל-1.37, והירידה המקסימלית קטנה מ-14.1% ל-11.7%.
  - התשואה השנתית יורדת מ-20.0% ל-16.6%.
- המשמעות: פוד ה-NDX גדול מדי בתיק. כל אסטרטגיה חדשה צריכה לנצח אג"ח קצרות באותו רבע, לא רק את G3.
- גם רגל הפריצה שעברה אתמול את הכלל (1.34) לא מנצחת אג"ח קצרות במקום הזה.

**מה בדקנו בשני מחקרים מוקפאים.**
- **ארביטראז' מיזוגים שמזוהה מהמחיר בלבד.** הרעיון: קפיצה עם נפח חריג, ואחריה מחיר "נעוץ" עד השלמת העסקה.
  - גרסה 1, על Russell 1000: מדויקת מאוד. 87% מהעסקאות הושלמו, ברווח ממוצע של 2.6% לעסקה. אבל היא תופסת רק כארבע
    עסקאות בשנה, ולכן כמעט תמיד במזומן.
  - גרסה 2, על Russell 3000 כולו: 34 עסקאות בשנה, רגישות לשוק כמעט אפס, מנצחת את קרן המיזוגים MNA ומנצחת מזומן.
  - אבל היא מרוויחה רק 2%-4% בשנה, ולכן בשנות הריבית הגבוהה (2022-2026) היא מפסידה לאג"ח קצרות בכל הגרסאות.
  - בנוסף, חצי אחוז טעות במחיר השלמת העסקה מוחק כחצי מהשארפ.
- **עונתיות חודשית:** מניות שעלו בחודש מסוים בשנים קודמות נוטות לעלות שוב באותו חודש.
  - הגרסה עם מסנן שוק קרסה ב-2022 (ירידה של 31%).
  - הגרסה המגודרת לא משאירה כמעט תשואה.

**התוצאה.**
- אף אסטרטגיה חדשה לא עוברת את הרף באופן שיש לו משמעות.
- חריג טכני אחד: גרסה 1 של ארביטראז' המיזוגים עוברת את הכלל בקריאה אחת של בדיקת החצאים, ונכשלת בקריאה השנייה.
  - גם כשהיא עוברת, היא מוסיפה לתיק רק 0.002 לשארפ.
  - היא מושקעת רק כ-3% מהזמן, כך שבפועל היא אג"ח קצרות עם שלוש עסקאות בשנה.
- יחד עם מחקר הטרנד והפריצה של הבוקר נבדקו היום שמונה משפחות, וכולן נכשלו: סטופים, פריצות, מומנטום שיורי, שתי גרסאות
  ארביטראז' מיזוגים, ושתי צורות של עונתיות.

**מה כן ממלא את הרבע הזה: אסטרטגיות שכבר יש לך.** זו בדיקה תיאורית ולא מוקפאת, על אותה היסטוריה שעליה נבנו.
- **זרימות סוף חודש (EOM flow)** בולטת: שארפ 1.08, 1.49 ו-1.52 בשלוש התקופות, מול 0.73, 1.33 ו-1.44 לאג"ח קצרות.
  - התיק מגיע לשארפ 1.49, תשואה 18.8% וירידה מקסימלית של 11.1%.
- **HPI:** שארפ 0.84, 1.42 ו-1.45. התיק מגיע ל-1.43, בלי לוותר על תשואה (20.5%).
- **הקפסולה** (DV2, HPI ו-ETF DV2) מגיעה ל-1.44.
- **אזהרה:** ב-EOM התזמון בסגירת המסחר נבחר בדיעבד. כשהפקודה מבוצעת בפתיחה הבאה, השארפ שלה לבד נמוך בהרבה
  (0.42-0.76).
  - היא גם עוד לא נבדקה בתוספת עלויות של 5 נקודות בסיס.
  - לכן זה כיוון, לא הוכחה.

**שורה תחתונה.**
- אין לי אסטרטגיה חדשה שכדאי לסחור בה. זה ממצא אמיתי ולא עצלות: החיפוש היה רחב, מוקפא מראש ונבדק פעמיים.
- הצעד הכי שווה עכשיו: לקדם את בדיקת הנייר של EOM flow בפקודות MOC, ואחריה את HPI או את הקפסולה. כל אחת נכנסת ברבע
  מהתיק, במקום חצי מפוד ה-NDX.
- עד אז: הקטנת פוד ה-NDX מ-50% ל-25% והחזקת אג"ח קצרות כבר עוברת את השער שלך (שארפ 1.37, ירידה 11.7%), במחיר תשואה
  נמוכה יותר.
- ארביטראז' המיזוגים אמיתי אבל קטן. כדאי להמשיך בו רק אם יהיה מקור נתונים של תנאי עסקאות.

## Bottom line (English)

- **The right hurdle is T-bills, not G3.**
  - Replacing a quarter of the book (half of the NDX pod) with T-bills (BIL) beats G3 in every block.
  - Sharpe by block: 0.728 / 1.334 / 1.437 against 0.689 / 1.303 / 1.257.
  - 2012-26: 1.365 against 1.288, drawdown -11.7% against -14.1%, CAGR 16.6% against 20.0%.
  - A new pod must beat T-bills in that slot. Yesterday's passing breakout leg (1.340) does not.
- **No new pod passes in any meaningful sense.** Across the two frozen studies here (36 + 14 configurations), no
  candidate beats the T-bill slot by a margin that matters.
  - The one formal exception is merger arbitrage v1's M1 cell. It passes only under the pooled reading of the R4 halves
    (amendment A1), and fails under the reading implemented.
  - Even when it passes it adds +0.002 Sharpe with about 3% exposure (Reality Check p = 0.92). It is T-bills plus three
    deals a year.
  - The individual families:
  - **Merger arbitrage v1 (Russell 1000):** precise but starved. Book +0.002 to +0.003 over T-bills, economically nil.
  - **Merger arbitrage v2 (Russell 3000):** real deal flow and near-zero beta, but only 2-4%/yr. Every cell loses to
    T-bills in 2022-26 (best -0.020), and capacity is below $5M.
  - **Seasonality:** the gated form loses -0.26 in 2022-26 (neighbourhood median), and the hedged form has no return
    left.
  - Reality Check p = 0.92 and 0.72.
- **What does beat T-bills in the slot is already in the owner's pipeline** (descriptive, section 5): EOM flow, HPI, and
  the MR capsule. They carry the usual in-sample caveats, and EOM's closing-auction timing was chosen after other
  timings had been seen.

## 1. The hurdle: trivial slot controls

Addition slot = TAA 0.5, L 0.25, slot 0.25. Official pod model (annual reset), each window its own run; Sharpe and max
drawdown are shown per window.

| Slot | 2008-11 | 2012-21 | 2022-26 | 2012-26 | 2008-26 | CAGR 2012-26 |
|---|---|---|---|---|---|---|
| none (G3 = TAA 0.5 + L 0.5) | 0.689 / -15.3% | 1.303 / -14.1% | 1.257 / -11.5% | 1.288 / -14.1% | 1.167 / -15.3% | 20.0% |
| cash at 0% | 0.722 | 1.325 | 1.357 | 1.334 / -11.7% | 1.201 / -14.5% | 16.2% |
| **BIL (T-bills)** | **0.728** | **1.334** | **1.437** | **1.365 / -11.7%** | **1.226 / -14.3%** | **16.6%** |
| SPY | 0.628 | 1.368 | 1.325 | 1.354 / -13.3% | 1.198 / -20.6% | 19.8% |
| MNA (merger-arbitrage ETF, from 2009-11) | – | 1.338 | 1.380 | 1.350 / -12.4% | – | 17.0% |

T-bills earned 0.3% / 0.5% / 3.9% a year in the three blocks. Even in the zero-rate years 2012-21 the T-bill slot beats
G3: the NDX pod adds little at a 50% weight.

## 2. Study 1 - merger arbitrage v1 and seasonality ([PREREG](NEW_POD_SEARCH_PREREG_20260927.md))

Margins are neighbourhood-median book Sharpe minus C_BIL, with the idle-cash sweep into BIL.

| Candidate (centre) | 2008-11 | 2012-21 | 2022-26 | DD gap 2012-26 / 2008-26 | R3 stress | R4 cross-check | R5 capacity | Verdict |
|---|---|---|---|---|---|---|---|---|
| M1 event x pin: J15 / theta 0.6% | +0.005 | +0.003 | +0.002 | 0.0 / +0.04 pp | pass | per-half REL25 (implemented): ex-S&P half 1.3650 vs 1.3652 **fail**; pooled REL25 (review re-run): 1.3674 / 1.3654 pass | pass | **fail / formal pass; economically nil** |
| M2 window x slots: anchor M0 | +0.004 | +0.005 | +0.003 | 0.0 / +0.10 pp | pass | ex-S&P half 1.363 **fail** | pass | fail |
| S GATED: SE_1_20 / N50 | +0.047 | +0.001 | **-0.260** | -0.97 / -1.39 pp | fail | fail | pass | fail |
| S HEDGED: SE_1_20 / N10 | +0.005 | -0.012 | **-0.122** | -0.19 / +0.03 pp | fail | fail | pass | fail |

**Merger arbitrage v1 (Russell 1000)**
- The detector is accurate but narrow:
  - 78 events a year pass the jump and volume tests, but only 2.4-7.4 pass the pin test;
  - the anchor made 3.8 entries a year and held 1.2 positions on average (6% exposure);
  - 87% of entries ended in a completed deal (+2.6% on average), while breaks lost -7.3%;
  - beta was 0.00.
- Its book is T-bills plus a few deals. The whole edge is about $12k of terminal P&L over 26 years; valuing deal
  completions at 0.99 of the last close removes it.

**Seasonality**
- The GATED cells have Sharpe 0.37-0.72 on their own, but a beta of about 0.45 and 15-18x turnover. The gate reopened
  into the 2022 bear market: -30.8% in 2022, against SPY -18%.
- After the SH hedge, the HEDGED cells have Sharpe between -0.66 and +0.24: the seasonal alpha is at best about 1%/yr.
- Month-end is the best of 21 rebalance days for the gated anchor, and at no day does either form beat C_BIL.

## 3. Study 2 - merger arbitrage v2, Russell 3000 ([PREREG](MERGER_ARB_V2_PREREG_20260927.md))

Version 2 was designed after seeing version 1's coverage diagnostics, and says so.

| Candidate (centre) | 2008-11 | 2012-21 | 2022-26 | DD gap | R3 | R4 halves: R1000 / R2000-only (vs 1.365) | R5 largest order at 5% of ADV | Verdict |
|---|---|---|---|---|---|---|---|---|
| Stage P: anchor V0 (J15, theta 0.5%, W5, K10) | -0.002 | -0.003 | **-0.037** | -0.21 / -0.28 pp | fail | 1.342 / 1.360 **fail** | $2.0M **fail** | fail |
| Stage S: W10 / K20 (grid corner) | +0.042 | +0.005 | **-0.030** | -0.18 / +0.08 pp | fail | 1.358 fail / 1.369 pass | $4.9M **fail** | fail |

**Diagnostics (anchor V0)**
- Flow and holdings:
  - 238 events, 38 confirmations and 33 entries a year;
  - 7.9 positions on average and 79% exposure;
  - median holding 43 sessions.
- Outcomes:
  - 77.7% of entries completed within a year;
  - completed deals averaged +1.9% (median +1.0%), and breaks -7.8%, filled 2.35% below the stop level;
  - beta 0.16, correlation 0.24 with TAA and 0.39 with L.
- The flow is mostly small caps:
  - R1000 events: 9.5 entries a year, precision 66%;
  - R2000 events: 27 entries a year, precision 79%.
- The stage-S centre makes 2.6%/yr at Sharpe 1.11 with a -6.2% drawdown and beta 0.05.

**Fragility**
- Terminal liquidations carry 134% of V0's P&L (breaks lose money).
- Valuing completions at 0.995 of the last close cuts the standalone Sharpe from 0.58 to 0.41; at 0.99, to 0.24.
- Without a deal-terms source, the result rests on the last traded price.

**Grid shape.** Standalone Sharpe rises monotonically with the pin window and the slot count, so the best cell is on the
grid boundary. The frozen plan did not extend the grid; that is a design note, not a result.

**What v2 does well.** It beats the MNA ETF slot and the 0% cash slot. The stream is statistically real (DSR 0.996 for
the stage-S centre). It is simply too small to beat 4-5% T-bills in 2022-26.

## 4. Checks

**Parity with the real engine** (replay of the replica's intents): every check reached daily-return correlation
1.0000000, with maximum daily differences of 4.4e-16 to 7.8e-16 and identical positions on every session.
- M0 on the Russell 1000.
- The seasonality GATED anchor and the HEDGED anchor (with SH) on the S&P 500.
- V0 on the Russell 3000: 876 traded names; final NAV identical to the cent.

**Download-date invariance**
- V1: decisions and positions are identical in every cell.
- Two edge cases:
  - one seasonality cell differs only through the engine's phantom $1 re-size fills (see the engine-issue task);
  - 4 of 12,658 v2 events sit exactly on the 10.0% threshold, and none confirms either way.
- V2: restating features to each decision day gives 0 flag mismatches.

**Other checks**
- Unit tests: 7 + 7 new tests pass.
- Memory refactor: the refactor done before any result was read reproduced the original run bit for bit.
- Independent review: see section 8.

## 5. What fills the slot today (descriptive; not pre-registered)

Same slot test, using the owner's existing pods. Series are from `portfolio_refresh_20260927` (corrected engine runs)
and `fund_product_menu_20260923/sources/eom_flow__path.csv.gz`.

| Slot | 2008-11 | 2012-21 | 2022-26 | 2012-26 Sharpe / MaxDD | 2008-26 Sharpe / MaxDD | CAGR 2012-26 |
|---|---|---|---|---|---|---|
| BIL (control) | 0.728 | 1.334 | 1.437 | 1.365 / -11.7% | 1.226 / -14.3% | 16.6% |
| CORE5 | 0.746 | 1.369 | 1.447 | 1.393 / -11.5% | 1.248 / -14.2% | 17.9% |
| DV2 | 0.806 | 1.448 | 1.382 | 1.426 / -13.0% | 1.277 / -14.8% | 21.7% |
| HPI (vote) | 0.844 | 1.417 | 1.454 | 1.428 / -11.0% | 1.301 / -15.7% | 20.5% |
| industry-ETF DV2 | 0.708 | 1.405 | 1.414 | 1.407 / -11.4% | 1.254 / -14.9% | 18.1% |
| MR capsule (DV2 / HPI / ETF DV2, 1/3 each) | 0.805 | 1.438 | 1.439 | 1.438 / -11.1% | 1.295 / -14.9% | 20.1% |
| **EOM flow** | **1.082** | **1.487** | **1.515** | **1.494 / -11.1%** | **1.408 / -11.1%** | **18.8%** |

Beats T-bills in all three blocks: EOM flow, HPI, CORE5 (narrowly) and the capsule (2022-26 by only 0.002). DV2 and the
industry-ETF DV2 each miss one block.

**Caveats that sit with this table**
- These pods were designed and tuned on the same history. None of this is pre-registered.
- EOM flow stands alone at 12.8%/yr, Sharpe 1.17 and -13.8% drawdown (2008-26), but:
  - its close-auction (MOC) timing was chosen after other timings had been seen;
  - its next-open versions reached only Sharpe 0.42-0.76 (`MR_BEYOND_DV2_20260926.md`);
  - 39% of its variance is the no-signal month-end Treasury cycle;
  - its small paper test is already the planned gate.
- The MR pods (DV2 / HPI) earn a liquidity premium paid mostly in stress, and their capacity is limited (MOC floors at
  about $10M per the growth shelf).

## 6. Recommendations

1. **Do not trade a new pod from these searches.** None beat T-bills in the slot.
2. **Move the EOM flow paper test (MOC execution) to the front.** It is the only candidate that clears the new bar by
   a wide margin. HPI and the capsule come next, subject to their known capacity limits and their pending paper test
   (industry-ETF DV2).
3. **Re-size the NDX pod as a book decision.** At 25% of the book with T-bills in the freed quarter, the book already
   meets the owner's gates: Sharpe 1.365, drawdown -11.7%. The trade-off is CAGR, 16.6% against 20.0%. This is a
   frozen-rule-free observation, but it holds in all three blocks and in both rate regimes.
4. **Merger arbitrage:** keep it as a research stream only. A deal-terms data source would be needed before a v3, which
   would test longer pin windows and more slots, where the v2 grid rose to its edge.

## 7. Limits

- One history; a 14-year book Sharpe has a standard error of about 0.27.
- G-P1 uses the synthetic TAA proxy.
- The idle-cash sweep is frictionless.
- The capacity figures are proxies.
- The slot tests use fixed 0.5 / 0.25 / 0.25 weights and are not optimised.

## 8. Independent review

**Reviewer:** an independent quant-pitfalls review (Opus 5.5, read-only) covered both studies and section 5.

**What it verified**
- No lookahead in either study.
- Terminal exits match Norgate's last dates (291 of 291 checked).
- Native Turnover is complete, and the parity replays are genuine.
- The rule arithmetic reproduces independently: C_BIL 0.7285 / 1.3341 / 1.4374, V0 -0.0016 / -0.0026 / -0.0373, and the
  stage-S centre +0.0417 / +0.0049 / -0.0296.
- Study 2's negative verdict is robust: every cell loses in 2022-26, and no cell can reach $5M capacity. That limit is
  structural: 10-20 slots of names near the REL25 cutoff.

**Findings, recorded as amendments A1-A4**
- **A1 (material to the formal verdict only).** The R4 halves of Study 1 recomputed the liquidity percentile inside each
  half. Under the pooled reading, merger arbitrage v1's M1 passes. Economically it is nil (section 2).
- **Minor.**
  - Family M's deflated Sharpe measured T-bill carry, not merger-arbitrage skill.
  - R5 was evaluated on the centre cell only; this has no effect.
  - The HEDGED form gives up part of the T-bill carry; it still fails when that is added back.

**On the section 5 table**
- The table reproduces exactly.
- EOM flow is causal and engine-costed. Its signal is measured seven sessions before month end, and its bucket uses
  prior months only.
- Caveats:
  - its closing-auction fills come from an adapter inside the strategy, not the engine's normal next-open path;
  - the timing was chosen after the next-open versions were seen;
  - it has not been run with the +5 bps stress;
  - negative cash (down to -9% of NAV) carries no financing charge.
- The existing pods in that table are unswept (idle cash at 0%), while BIL earns T-bill returns. The table therefore
  understates them slightly against the control.
- Treat the EOM row as in-sample and timing-selected.
