# What belongs in the fund's Growth products (2026-11-01)

**Question (owner):** what is worth putting into the fund's Growth products? Include the ADV version found for DV2. Judge at portfolio level, not each strategy alone. Mixing is always allowed.

**Answer:**

- **Same engines, one change: DV2 and HPI with the gate, each entry sized by the stock's liquidity.** At the same weights every product gains about 1 point of CAGR and 0.04 to 0.05 of Sharpe in the backtest. About half of that is simply bigger entries, and half is liquidity information that rests on a few years, so plan on about half the gain. What does not depend on history: orders in thin names shrink, and the book holds to $5M at the open instead of $2.5M. The signal, gate and exit rules stay as wired. The names held do not stay the same: larger entries fill the book sooner, so there are fewer entries and about a third of entry days buy a different set of names.
- **The ADV version of DV2 does not earn a place by itself.** It has lower Sharpe than today's form in the book, and sizing reaches the same capacity at higher Sharpe.
- **Month-end flow is the only other candidate that improves the book.** It lifts Sharpe by about 0.1, but it is weak in 2026 and not wired yet. Every other candidate fails or only lowers risk, which the TAA weight does anyway.
- **MR stays at 30%. TAA stays TAA 1N.** Moving along the products is a dial between TAA and CORE5.

Research only. No live path, strategy, release or portfolio file changed. Frozen plan and amendments 1 to 4: `scripts/research/growth_mr_products_20261101/`, ledger row 559. Outputs: `results/scout/growth_mr_products_20261101/`. Independent review: `review/REVIEW.md` in that folder.

All numbers: 2008-03-04 to 2026-08-19, gross of fund fees, idle cash at a fair rate, annual reset. Sharpe is over T-bills. Before 2012-10 TAA is a synthetic proxy.

## The menu

"Sized" is DV2 and HPI at 15% each, VIX gate, BIL parking, liquidity sizing. The right column is the same weights with today's 1/N capsule.

| Product | TAA 1N | MR sized | CORE5 | Month-end | CAGR | Sharpe | Max DD | P(DD below -20%) | Today's capsule, same weights |
|---|---|---|---|---|---|---|---|---|---|
| Growth | 40 | 30 | 30 | | 19.8% | 1.33 | -12.1% | 3% | 18.7% / 1.28 / -13.1% |
| Growth Plus | 50 | 30 | 20 | | 21.9% | 1.30 | -14.0% | 11% | 20.9% / 1.26 / -15.6% |
| Aggressive | 60 | 30 | 10 | | 24.0% | 1.28 | -16.6% | 29% | 22.9% / 1.24 / -18.1% |
| Growth, with month-end | 40 | 30 | 15 | 15 | 20.9% | 1.43 | -12.2% | 2% | 19.9% / 1.38 / -12.9% |
| Growth Plus, with month-end | 50 | 30 | 0 | 20 | 23.5% | 1.41 | -13.9% | 7% | 22.5% / 1.36 / -14.5% |
| Aggressive, with month-end | 60 | 30 | 0 | 10 | 24.8% | 1.32 | -15.1% | 23% | 23.8% / 1.28 / -16.5% |

| Product | +5 bps | Planning case | Net of 1 / 15 | P(18-year CAGR >= 20%) | Alpha, gross | Alpha, net | Alpha 2022+, gross | Alpha 2022+, net |
|---|---|---|---|---|---|---|---|---|
| Growth | 18.5% / 1.25 | 13.6% / 0.94 | 15.9% / 1.06 | 46% | 8.1% (t 4.9) | 4.8% (t 2.7) | 8.1% (t 2.8) | 4.5% (t 1.4) |
| Growth Plus | 20.6% / 1.24 | 15.1% / 0.92 | 17.7% / 1.05 | 71% | 8.8% (t 4.6) | 5.3% (t 2.6) | 9.0% (t 2.7) | 5.2% (t 1.4) |
| Aggressive | 22.7% / 1.22 | 16.5% / 0.91 | 19.4% / 1.04 | 85% | 9.4% (t 4.4) | 5.7% (t 2.6) | 9.9% (t 2.6) | 5.9% (t 1.4) |
| Growth, with month-end | 19.4% / 1.33 | 14.2% / 0.99 | 16.8% / 1.14 | 62% | 9.3% (t 5.6) | 5.9% (t 3.3) | 9.1% (t 3.1) | 5.4% (t 1.7) |
| Growth Plus, with month-end | 21.8% / 1.31 | 15.9% / 0.98 | 19.0% / 1.13 | 86% | 10.3% (t 5.4) | 6.6% (t 3.3) | 10.4% (t 3.0) | 6.5% (t 1.7) |
| Today's Growth Plus, for reference | 19.7% / 1.19 | 14.4% / 0.89 | 16.8% / 1.01 | 60% | 8.0% (t 4.4) | 4.6% (t 2.4) | 7.9% (t 2.4) | 4.2% (t 1.2) |

Planning case: every pod keeps 0.75 of its past excess, +5 bps. Alpha: over the ETF mix + QQQ trend, weekly, Newey-West.

## The MR slot, in the Growth Plus book

| MR form | Alone: CAGR / Sharpe / Max DD | Book CAGR | Book Sharpe | Book Max DD | Beats today, share of paths | Holds at the open |
|---|---|---|---|---|---|---|
| Today: gate, 1/N | 16.8% / 1.02 / -21.4% | 20.9% | 1.26 | -15.6% | | $2.5M |
| **Gate, liquidity sizing** | **19.9% / 1.14 / -18.5%** | **21.9%** | **1.30** | **-14.0%** | **1.00** | **$5M** |
| DV2 sized, HPI with floor and no gate (the frozen rule's pick) | 19.8% / 1.15 / -17.3% | 21.7% | 1.28 | -13.6% | 0.84 | $5M |
| DV2 sized alone | 22.4% / 1.08 / -24.0% | 22.7% | 1.31 | -14.6% | 0.96 | $2.5M |
| ADV DV2 with gate + HPI with gate | 14.6% / 0.97 / -17.5% | 20.2% | 1.24 | -13.4% | 0.15 | $5M |
| ADV DV2 with gate, alone | 14.4% / 0.88 / -21.0% | 20.2% | 1.22 | -13.9% | 0.15 | $5M |
| ADV DV2 no gate, alone | 17.3% / 0.91 / -22.4% | 20.9% | 1.19 | -14.0% | 0.09 | $2.5M |
| Half ADV, half sized DV2, + HPI | 17.3% / 1.05 / -19.6% | 21.1% | 1.27 | -13.5% | 0.74 | |
| Floors + sizing on both, gate | 16.5% / 1.05 / -15.5% | 20.8% | 1.27 | -13.1% | 0.61 | $5M |
| No gate, 1/N | 18.6% / 1.00 / -23.1% | 21.3% | 1.22 | -15.3% | 0.15 | |
| No gate, sized | 21.7% / 1.10 / -19.5% | 22.3% | 1.26 | -13.9% | 0.48 | |
| Capsule + DV2-IND, thirds | 13.7% / 1.09 / -15.1% | 19.9% | 1.25 | -14.5% | 0.24 | |

Reading:

- **Dropping the gate loses in the book.** Alone the ungated forms look strong, but they hold stocks 77% of the time against 39%, and their correlation with TAA is 0.53 against 0.41.
- **Choosing liquid names is not the same as sizing up liquid names.** The floor throws away the volatile names where reversal pays. Sizing keeps the same signal and rank rules and moves dollars between the names. ADV gives up Sharpe, sizing gains it, and both reach about the same capacity.
- **The rules are the same, the holdings are not.** Larger entries fill the book sooner. On the runs to 2026-10-30: DV2 4,974 entries against 4,656, HPI 3,026 against 2,556, and 32% and 31% of entry days buy a different set of names. So the sized pod cannot be read as the 1/N pod with other weights.
- **My prior was half wrong.** I expected the floor forms to gain alone and lose in the portfolio. With the gate, ADV DV2 also loses alone (Sharpe 0.88 against 0.96).

## How sure is the sizing gain

The independent review found that the sizing rule does two things at once. It is not normalised, so real entries are larger than 1/N's. It also moves dollars toward liquid names. Two forms separate the parts, both after the result, using constants taken from the rule's own entries, so they are a decomposition and not candidates.

| Form, Growth Plus weights | CAGR | Sharpe | Max DD | Max DD on real TAA data (from 2012-10) | Beats 1/N, share | Gain against 1/N: t, years up / down | Holds at the open | 1-in-100 order is 5% of a day at |
|---|---|---|---|---|---|---|---|---|
| 1/N: entries 10% of the pod | 20.9% | 1.26 | -15.6% | -14.2% | | | $2.5M | $192M |
| Bigger entries only: 10.7% in DV2, 12.1% in HPI, no liquidity input | 21.4% | 1.28 | -14.5% | -14.5% | 0.99 | t 3.2, 14 / 2 | $2.5M | $166M |
| Liquidity only: entries average about 9.6% | 21.1% | 1.28 | -13.7% | -13.4% | 0.87 | t 0.9, 8 / 5 | $5M | $425M |
| **Liquidity sizing, both** | **21.9%** | **1.30** | **-14.0%** | **-13.6%** | **1.00** | **t 3.7, 13 / 3** | **$5M** | **$353M** |

- **Bigger entries are the steady part.** Of +1.01 points of CAGR, +0.58 comes from size alone; of +0.046 of Sharpe, +0.022. It holds without any single good period. A constant anywhere from 1.05 to 1.5 raises the book's Sharpe smoothly, from 1.26 to 1.29, so it is not a tuned level. It is not the same as a bigger slot: 1/N with MR at 35% and CORE5 at 15% is 21.4% / 1.26 / -16.5%. But it pushes more dollars into thin names, and on real TAA data its Max DD is worse than 1/N's.
- **Liquidity is the less steady part.** Sizing over bigger entries only: +0.43 points, +0.024 of Sharpe, share 0.95, t 1.7.

| Sizing over bigger entries only, years left out | Sharpe | CAGR, points | t |
|---|---|---|---|
| None | +0.024 | +0.43 | 1.7 |
| 2020 | +0.012 | +0.20 | 0.8 |
| 2020, 2025 | +0.007 | +0.10 | 0.4 |
| 2020, 2024-26 | 0.000 | -0.06 | -0.3 |

Its worst case in this table is flat, not a loss. On 1/N's own trades, size and trade return still correlate month by month (0.078, t 4.6 in DV2; 0.072, t 3.4 in HPI), and the capacity gain is mechanical.

- **The whole gain has a few big years too.** 2020 is about a third of the sum of the yearly gaps, and 2020 with 2025 about half. Against 1/N: without 2020 +0.034 of Sharpe, +0.74 points, t 2.9; without 2020-21 and 2023-26 +0.022, +0.40, t 1.6; 2010-19 only +0.014, +0.29, t 1.4.
- **Most of the Max DD gain leans on the 2008 proxy.** The book's worst fall runs 2008-06 to 2009-05, before TAA's real data starts. The 1.6 points become 0.6 on real data. Alone, the sized capsule falls further than 1/N in 2012-19, 2022 and 2023-26 (-12.6%, -15.7%, -14.8% against -11.1%, -13.1%, -12.1%).

| Other tests | Result | Read |
|---|---|---|
| All adoption checks against today (rung, share, breach, exact window, +5 bps, both halves) | all pass, share 1.00, gap 5 / 50 / 95: +0.02 / +0.05 / +0.08 | conditional on the form having been chosen; shares in other books are the same active return, not new evidence |
| 2022-26 block | 1.27 against 1.22 | holds |
| Selection | across the 13 sizing rules, gated, in the book: p 0.06. Among all 3,213 sleeves: 0.10 studentised, 0.60 on the frozen statistic, best sleeve 0.48 | quote about 0.06 to 0.10. Not proof |
| Chosen on the first half | picks this rule; second half 1.47 against 1.41 | supports it |
| Random sizes, 300 draws from the rule's real entries | real 1.30, median 1.27, 95th 1.29: above all 300 | above luck by 0.03 of Sharpe and 0.6 points |
| Mirror, the same spread the other way | at its own size it loses 0.028 | direction matters |
| Without each pod's 10 best and 10 worst names | keeps about half the gain | not a few names |
| Size known 1, 21, 63, 252, 504 sessions late | 1.301, 1.298, 1.293, 1.271, 1.268 | decays within a year, so not a fixed large-cap tilt |
| Market exposure | active beta to SPY 0.004, since 2022 0.012; mean stock exposure unchanged (41.2% against 41.6% in DV2, 36.5% against 36.0% in HPI) | not more market |
| Costs | gap at +5, +10, +20, +50 bps: 0.97, 0.92, 0.84, 0.64 points; notional in the least liquid third falls from 33% to about 22% | not eaten by costs |

A forward test takes years. The book's gain is 0.85 points a year on 1% of tracking error, so even if the past gain holds it needs about 5 years to reach t 2.

## Where the frozen rules ended, and where I depart from them

| Stage | The rule's result | What I recommend | Why |
|---|---|---|---|
| Which MR form | DV2 sized + HPI with floor and no gate. It passes adoption. | DV2 and HPI both gated and sized. A judgement after the result. | Same plateau band. The rule's capacity tie-break, decided by 0.4%, cannot see the rest: the pick needs a new filter, drops the gate on half the sleeve, trades 961 orders a year against 699. Head to head Sharpe share 0.84, CAGR share 0.64; 2022-26 1.27 against 1.22. Cost of my choice: Max DD -14.0% against -13.6%, first block 0.94 against 0.95. |
| Weights | One book for every rung: TAA 40 / MR 30 / CORE5 30. Higher rungs not offered. | That book is Growth. Keep today's 50 and 60 TAA as Growth Plus and Aggressive. | The rule keeps books within 0.03 of the best Sharpe. Sharpe falls 0.03 per 10 points of TAA, so the band alone sets the risk (band 0.01 gives TAA 35, 0.05 gives 50). It is not evidence against higher rungs. The price of each step is on the line below. |
| MR above 30% | Does not move. The challenge fails on share (0.67-0.72), breach and +5 bps. | Stays 30%. | Agrees. |
| TAA form | Half 1N, half classic passes (share 0.94); classic only passes (0.87). TAA 2x fails. | Stays 1N. This departs from the frozen text (`AMENDMENT_4.md`). | The checks read Sharpe only. The blends gain through lower volatility, but at the same Max DD they earn less: 1N at 50 is 21.9% / -14.0%, half and half at 55 is 21.0% / -14.7%. Classic also holds more BTAL, which binds at the auctions. |
| Other candidates | Month-end passes in all 4 forms. DV2-IND, the pair and downshock pass in some forms. | Month-end only. | See below. |

**The dial, with sized MR at 30%:**

| TAA / CORE5 | CAGR | Sharpe | Max DD | P(DD below -20%) | TAA share of risk | Sharpe if TAA's edge is gone |
|---|---|---|---|---|---|---|
| 40 / 30 | 19.8% | 1.33 | -12.1% | 3% | 67% | 0.57 |
| 45 / 25 | 20.8% | 1.32 | -12.9% | 6% | 71% | 0.51 |
| 50 / 20 | 21.9% | 1.30 | -14.0% | 11% | 75% | 0.46 |
| 55 / 15 | 22.9% | 1.29 | -15.3% | 19% | 78% | 0.42 |
| 60 / 10 | 24.0% | 1.28 | -16.6% | 29% | 80% | 0.38 |
| 65 / 5 | 25.0% | 1.26 | -17.8% | 41% | 83% | 0.35 |

More return means leaning harder on one engine. If MR's edge is gone, the Growth Plus book falls to 15.8% / 0.96; TAA 60 / CORE5 40 with no MR is 19.8% / 1.15.

## Anything else in the book

Tested in the Growth Plus book at 10% and 20%, taken pro rata from everything or from CORE5 first. Base 21.9% / 1.30 / -14.0%. Alpha is weekly; the frozen text said daily, and nothing gates on it.

| Candidate | Alone | Alpha over the book | Taken from CORE5, 10% | Pro rata, 10% | Verdict |
|---|---|---|---|---|---|
| **Month-end flow** | 13.9% / 1.14 | 10.3% (t 4.3) | 22.7% / 1.36 / -13.5%, share 1.00 | 21.2% / 1.37 / -12.2%, share 1.00 | **In, once wired** |
| DV2-IND | 7.3% / 0.96 | 2.9% (t 2.4) | 21.9% / 1.31 / -14.4%, share 0.66, fails | 20.5% / 1.32 / -12.8%, share 0.95 | No gain over moving the dial |
| QQQ-IWM pair | 9.7% / 0.67 | 5.2% (t 1.6) | 22.2% / 1.32 / -13.5%, share 0.78, fails | 20.7% / 1.33 / -12.0%, share 0.89 | Small gain, not wired, not worth a build here |
| Downshock | 8.9% / 0.85 | 2.2% (t 1.4) | 22.1% / 1.30 / -14.5%, fails | 20.6% / 1.31 / -12.8%, share 0.84 | No gain over moving the dial |
| Inflation Compass QQQ | 18.6% / 0.95 | 2.9% (t 0.8) | fails | fails | Out |
| BTAL_QQQ | 12.5% / 1.13 | 0.5% (t 0.4) | fails | fails | Out |
| Sector dispersion | 7.9% / 0.90 | 1.3% (t 1.0) | fails | fails | Out |
| TAA linearity QQQ | 12.2% / 1.06 | 0.1% (t 0.1) | fails | fails | Out |
| Gated QQQ | 11.8% / 0.76 | -2.9% (t -1.2) | fails | fails | Out |
| Closed momentum capsule | 16.2% / 0.89 | 0.8% (t 0.3) | fails | fails | Out, as decided |

Moving the dial to TAA 45 / CORE5 25 gives 20.8% / 1.32 / -12.9%. A pro-rata 10% of DV2-IND or downshock lands on the same point, so passing the test adds nothing there. The test reads Sharpe only: 20% pro rata in DV2-IND passes while CAGR falls by 3 points. The rule's single book (TAA 40) gives the same picture, except that the pair also passes 10% from CORE5 there (share 0.81).

**Month-end, with its caveats:** it has almost no overlap with the book (correlation 0.14). But alone it is -3.1% in 2026 to 2026-10-14 with T-bills at +2.9%, and its net alpha since 2023 is not significant (`EOM_TRADABLE_20261101`). The plan number for it alone is Sharpe 0.95, not 1.26. If its edge is gone, the Growth Plus book with it falls to 1.25 and 20.8%, against 1.30 and 21.9% without it. The sleeve holds to about $5-10M, so a 20% share caps the book at $25-50M. The new vote rule ties the old rule inside these books (23.5% / 1.40 / -13.4%), so the numbers above carry over. It is the form that is PM_READY for wiring.

## Stress

| Fall | Today's Growth Plus | Sized Growth | Sized Growth Plus | Sized Aggressive | Sized Growth Plus, with month-end |
|---|---|---|---|---|---|
| 2008 crisis | -10.3% | -6.7% | -8.8% | -11.0% | -6.3% |
| Q4 2018 | -6.0% | -5.4% | -6.2% | -6.9% | -5.2% |
| Covid | -13.2% | -10.8% | -12.3% | -13.8% | -12.3% |
| 2022 bear | -7.9% | -5.6% | -7.8% | -9.9% | -5.3% |
| 2025 tariffs | -10.6% | -8.7% | -9.7% | -10.6% | -8.4% |
| Worst calendar year | +0.6% (2022) | +1.7% (2022) | +0.3% (2022) | -1.1% (2022) | +2.8% (2022) |

## Capacity

Route model, orders from 2023-08-21, pre-TCA. Largest size where every gate holds and cost stays under a quarter of the excess return. Sizes are from a coarse grid: $2.5M, $5M, $10M, $25M, $50M, $100M, $250M. The pods' returns are open fills, so **only the open column matches the backtest**. The last column sends MR's stock orders to the close, a route with no backtest behind it.

| Book | Market on open | Everything at the close | Worked ETFs, MR stocks at the close | What breaks that last route next |
|---|---|---|---|---|
| Today's capsule | $2.5M | $5M | $10M | FOX at $25M |
| Bigger entries only | $2.5M | $5M | $10M | FOX at $25M |
| Liquidity only | $5M | $5M | $25M | FOX at $50M |
| Sized, any of the three products | $5M | $5M | $25M | FOX at $50M |
| ADV DV2 with gate, alone | $5M | $5M | $25M | TER at $50M |
| ADV DV2 no gate, alone | $2.5M | $5M | $10M | EQT at $25M |
| Floors + sizing | $5M | $5M | $50M | AJG at $100M |
| No MR: TAA 60 / CORE5 40 | $5M | $5M | above $250M | |

BTAL inside TAA sets a $10M wall at both auctions in every book. Sizing lifts the stock limit by shrinking orders in thin names. With $100M in the book, the 1-in-100 stock order falls from 2.6% to 1.4% of a median day, and the largest from 7.6% to 3.8%. Orders a year fall from 1,092 to 996; stock turnover rises from 11.7 to 12.5 times the book. With Nasdaq 100 names flagged in the model, the sized book's small cost edge at the open turns into about 1 bp the other way, and no size moves.

If the fund outgrows the open route, floors + sizing is the next form to study, with its own test of a close or worked route for MR. It costs about 1 point of CAGR and 0.04 of Sharpe against the sized capsule. Above that, MR must shrink.

**Added 2026-11-02, after the build: the Bench capacity engine, per pod, at the open.** The table above answers "where does a gate break", with a bar of cost under a quarter of the excess return. It hides what the size costs. The engine's central estimate, last 5 years, pre-TCA, MR orders only, with 30% of the book in the capsule:

| Book | Per pod | 1/N capsule | Sized capsule | Cost in the book, either form |
|---|---|---|---|---|
| Before cost | | 16.6% | 22.1% | |
| $1.67M | $250K | 14.6% | 20.0% | 0.6 points a year |
| $3.3M | $500K | 13.0% | 18.3% | 1.1 |
| $6.7M | $1M | 10.6% | 15.7% | 1.9 |
| $13.3M | $2M | 6.6% | 11.5% | 3.2 |
| $33M | $5M | -4.2% | 0.0% | 6.6 |

The engine's own recommended size is $250K for sized DV2 against $100K at 1/N, and $100K for HPI in both forms; the break-even bracket is $2M to $5M against $0.5M to $1M in DV2, and $1M to $2M against $0.5M to $1M in HPI. So the form choice stands: the cost is nearly equal in both, and sizing leads at every size. What changes is the fund's expectation: at $5M of book about 1.5 points a year below the gross numbers of this record, and more above it. This window's lead before cost, 5.6 points, flatters; over the whole history it is 2.6.

The close route is now tested too, and it does not rescue the size. In Bench Timing, entries and exits at the next close take sized DV2 from 19.2% / 1.06 / -23.9% to 11.6% / 0.71 / -38.7%, and sized HPI from 14.6% / 1.05 / -19.8% to 7.2% / 0.59 / -35.6%. Do not quote the $25M and $10M columns above. Evidence: `results/research/mr_capsule_liq_build_20261101/capacity_compare.json` and the pods' folders under `results/research/strategy/`.

## After the Lab's end

From 2026-08-20 to 2026-10-14, 39 sessions, fresh runs of every pod, T-bills +0.6%:

| | Since Aug 19 | 2026 to Oct 14 |
|---|---|---|
| Today's Growth Plus | +0.8% | +25.8% |
| Sized Growth Plus | +0.7% | +27.7% |
| Sized Growth Plus, with month-end | +0.2% | +24.6% |
| TAA 1N | +0.8% | +37.9% |
| CORE5 | +1.2% | +12.2% |
| DV2 gate, 1/N against sized | +3.5% / +5.3% | +20.5% / +28.1% |
| HPI gate, 1/N against sized | -2.7% / -5.2% | +9.3% / +13.8% |
| Month-end, old rule / vote rule | -2.2% / -2.2% | -3.1% / -2.1% |

A tie since the Lab's end, in a window too short to say more. 2026's book return is mostly TAA.

## Independent review

A separate agent read the code and rebuilt the numbers from the raw files, without seeing this record.

- **Clean:** no look-ahead in the size matrices (truncation test at 4 dates, gap 0), no split leak in the liquidity data, entry sizes share-exact on the eight sized pods, BIL orders correct on the true dollars, negative-cash sessions 55 and 50 against 167 and 168. The book rebuilt from raw files matches to under 0.00001. All 3,213 sleeves, the Stage 2 grid, the Stage 3 legs and 24 lens books match.
- **Changed the reading:** the split above, the selection numbers, the Max DD on real data, the concentration by year, and the capacity route. They are in the sections above.
- **Corrected:** `AMENDMENT_3.md` said the sized capsule has the higher Sharpe in every block than the rule's pick; block A is 0.940 against 0.949. `falsify.py` drew random sizes from all candidates; the numbers above use the review's draws from real entries. Small items, none affecting a number here, are in the review file.
- **Only read, not rebuilt:** engine internals, the Lab's TAA and CORE5 columns, the 2008 proxy, the alpha tables, and the route figures for books other than today's and the sized Growth Plus.

## Recommendation and its price

Adopt the gated capsule with liquidity sizing as the MR form of the Growth products. Plan on about half the backtest gain: +0.5 points of CAGR, +0.02 of Sharpe, and $5M at the open against $2.5M. Treat the rest as upside.

The price is a build, and the owner chose it on 2026-11-01. Real modules, registry entries, replay parity and PM checks are done, with the templates disabled; the record is `docs/live/MR_CAPSULE_LIQ_SIZING_WIRING_20261101.md`. NDX and TAA paths are not touched.

Correction from the build review: this section first said one could go live at 1/N and switch later at the cost of a parity run. That is wrong. A pod cannot move between the two forms: the account must be flat, BIL included, and restart under a new pod id. So the form should be chosen before go-live. Bigger entries alone are simpler, but they raise orders in thin names, and sizing beats them on every number here.

## Limits

- All of this history has been seen, and the forms were shaped on it.
- MR has no live fills. Flat costs inside the Lab; the route model is pre-TCA.
- The sized and floor pods are research subclasses, on the frozen sign-off inputs to 2026-10-14.
- The stock cap is 99% of NAV, set after the first runs failed a cash check (`AMENDMENT_2.md`). A single name can reach about 24% of a pod with sizing, against about 13% at 1/N.
- HPI's existing relative-liquidity module computes dollar volume with the unit the 2026-09-26 audit invalidated. It is research-only and not used here; the floor uses native Turnover.
- `MR_POSITION_SIZING_20261101` quoted gated numbers with a BIL parking defect. They are corrected there; the fix moved the book by 0.01 of Sharpe.
