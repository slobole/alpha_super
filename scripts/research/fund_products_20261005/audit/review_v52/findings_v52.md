# REVIEWER: recompute
VERDICT: No blocker. Every number I rebuilt with my own book arithmetic for Growth Plus, Monthly and Monthly Plus matches the JSON files, the page and the record to 4 decimals. One major disclosure gap (the monthly books' rung label holds only near the full edge and the page says "not computed") and five minor items.

Scratch scripts are in C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf\scripts\research\fund_products_20261005\audit\review_v52\ (rc1.py, rc2.py, rc3.py, rc1_out.json). No study file was edited; nothing was published or committed.

## recompute F1 [major] fund_products_v5.html, table "DD beyond the product's limit", column "Edge margin (lowest k that passes)", rows Monthly and Monthly Plus = "not computed"; cards Monthly / Monthly Plus "Rung passed GROWTH" / "GROWTH PLUS" with no edge qualifier; record table rows "Monthly (runs first, scales) ... GROWTH" and "Monthly Plus ... GROWTH PLUS"; battery.py edge_margin loop (products only)
PROBLEM: The two monthly books carry a rung label without the edge margin that every capsule product shows, and that margin is much thinner than for the capsule products. Monthly holds its GROWTH label only down to 0.95 of the edge and Monthly Plus holds GROWTH PLUS only down to 0.90, against 0.70 for Growth, 0.75 for Growth Plus and 0.85 for Aggressive (which the page calls out as weak). Under the full rung rule (worst seed and +5 bps as well) Monthly already fails at 0.95. The book described as "runs first" therefore passes its rung essentially only at the full backtest edge, and the reader is told "not computed".
EVIDENCE: Own computation, study convention (MAIN frame, 10-seed mean, cap 15%). Monthly at -20%: k=1.00 11.2%, 0.95 13.7%, 0.90 16.7%, 0.85 20.5%, 0.75 30.0%. Monthly Plus at -25%: k=1.00 9.6%, 0.95 11.9%, 0.90 14.7%, 0.85 18.0%, 0.75 27.3%. Same code reproduces the study's margins for Growth (13.4% at 0.70) and Growth Plus (12.3% at 0.75, 16.0% at 0.70). Full rule at k=0.95: Monthly +5 bps worst seed 16.05% (fails), +5 bps mean 14.9%; Monthly Plus passes at 0.95 (worst 14.2%) and fails at 0.90 (MAIN worst seed 16.5%).
FIX: In battery.py add "S9 incumbent launch" (limit -20%) and "old growth plus" (limit -25%) to the edge-margin curve. Print 0.95 and 0.90 in the table instead of "not computed", add "holds down to 0.95 / 0.90 of the edge" to the two cards, and add one sentence to the record's monthly section and caveat table.

## recompute F2 [minor] fund_products_v5.html card Monthly Plus "Capacity, worked ≥$250M", menu table column "Capacity (worked route, upper bound)" = "≥$250M", capacity table row Monthly Plus; record capacity table row "Monthly Plus | $2.5M | $2.5M | $250M or more | ... | $217.7M"
PROBLEM: For Monthly Plus the worked-route figure is printed as a floor ("$250M or more") while the same row shows a BTAL 10%-ownership wall of $217.7M, which is below it. The two numbers contradict each other on the new 65 / 35 weights. Monthly is unaffected (wall $283M).
EVIDENCE: capacity.json "old growth plus": routes["worked+blocks"].recommended = 250,000,000 (grid top); btal_wall = 0.10 x 317e6 / (0.65 x 0.224) = 217,719,780.
FIX: Show the worked capacity as min(route figure, BTAL wall), i.e. about $218M for Monthly Plus, or drop the "≥" and add "BTAL wall $218M binds first".

## recompute F3 [minor] portfolios/fund_growth.yaml, fund_growth_plus.yaml, fund_growth_aggressive.yaml, header line "# Rung: target ...; strictest passed .... Plan on the planning column of the report, not on these numbers." (written by pm_confirm.py write())
PROBLEM: Stale text: the planning column was removed from the report, so the YAML header points to a column that no longer exists.
EVIDENCE: portfolios/fund_growth_plus.yaml line 4; the page and record have no planning column (record line 7: replaced by one conservative and one stress case).
FIX: Change the f-string in pm_confirm.py to refer to the conservative case in the "how much to believe" section and rewrite the three YAML headers. This is a comment-only change, so no engine re-run is needed.

## recompute F4 [minor] fund_products_v5.html blend section, "CAGR −0.5 … +0.1 pp"; record line 68 "the blend's CAGR differs by -0.5 to +0.1 pp"
PROBLEM: The upper end of the range is a rounding artefact. The blend never beats cash dilution by 0.1 pp of CAGR; the largest gap is +0.05 pp.
EVIDENCE: Own computation, blend minus Growth diluted with BIL to the same volatility: 20% -0.47 pp, 40% +0.05 pp, 60% +0.05 pp, 80% +0.03 pp. The Sharpe range (-0.01 to +0.03) and the Max DD range (+0.7 to +1.4 pp) are correct.
FIX: Print "−0.5 … +0.05 pp" with two decimals, or say "equal within 0.05 pp at 40 / 60 / 80%, 0.5 pp lower at 20%".

## recompute F5 [minor] g_lib.py "RUNG_EXEMPT = {S9, S10}"; page challenger table row "S9 Monthly (TAA 3x 1N 50 / CORE5 50)", column "0 rung" = "exempt (passes)"
PROBLEM: The exemption from check 0 was registered for the incumbent of 2026-10-01. After the key re-use, "S9 incumbent launch" is the new two-pod Monthly, designed after results, so the exemption now sits on the wrong book; the real incumbent (S13) is tested normally. No number changes, because both pass GROWTH.
EVIDENCE: study.json challenges: S9 c0_rung true (exempt); S13 passes the rung on its own. Monthly's rung detail: Max DD -13.9% / -14.0%, breach at -20% 11.2% (worst seed 12.45%), at +5 bps 12.25% (worst seed 13.35%).
FIX: Set RUNG_EXEMPT = {S13_OLD, S10} and print "yes" for S9, or leave the code and change the cell to "yes".

## recompute F6 [minor] fund_products_v5.html, monthly section, bullet "למה צריך את CORE5: בלי רגל הגנתית (TAA 3x 57 / momentum 43) התיק נכשל במדרגת GROWTH: Max DD −16.9%, DD beyond −20% 17.5%"; monthly.py reference row "TAA 3x 57 / momentum 43, no CORE5"
PROBLEM: The evidence for "why CORE5 is needed" uses a different book from the one sold: TAA 3x (not 1N) plus momentum, with no CORE5. The Monthly books hold TAA 3x 1N and no momentum. Printing "Max DD −16.9%" beside "fails" also reads as if the drawdown fails, when it passes the -17% build limit and only the breach figure fails.
EVIDENCE: monthly.json reference: weights taa3x 0.571 / MOM 0.429, dd -0.1685 (passes -17%), p20 0.1754 (fails the 15% cap). The like-for-like figure exists on the page: TAA 3x 1N alone Max DD -26.7%, breach at -20% 98.5%.
FIX: Quote TAA 3x 1N alone (or add a "1N 100" ladder row) as the no-CORE5 evidence, and say "fails on the breach figure (17.5% against the 15% cap)".

# REVIEWER: stale
VERDICT: No blocker. The page (v5.2) and the record are numerically consistent with the JSON files for the current product set; every figure I recomputed matched. I found 1 major issue (a rung verdict on the two monthly cards that lacks a qualifier the page's own data require) and 17 minor wording, labelling and readability issues. No leftover "Planning"/"Floor" wording or old planning numbers, no "one third each" applied to Growth Plus, and no four-pod description applied to the new Monthly.

## stale F1 [major] fund_products_v5.html, cards "החודשי Monthly" and "החודשי פלוס Monthly Plus": row "Rung passed GROWTH" / "Rung passed GROWTH PLUS"; also the RUNG details table, column "Edge margin (lowest k that passes)" = "not computed" for both
PROBLEM: The capsule cards qualify the rung verdict ("holds down to 0.70 / 0.75 / 0.85 of the edge"); the monthly cards give an unqualified pass. The page's own conservative case shows both monthly books fail their rung's 15% breach cap at 3/4 of the edge, and nothing says so in words. The product that "runs first" therefore reads as holding its rung as robustly as Growth, which it does not.
EVIDENCE: battery.json edge_decay.scenarios["all at 0.75"]: Monthly p20 = 30.0% (cap 15%), Monthly Plus p25 = 27.3% (cap 15%). Growth p20 = 10.2% and Growth Plus p25 = 12.3% pass; Aggressive 19.8% fails and that is stated. At full edge Monthly is 11.2% (worst seed 12.45%, at +5 bps 13.35%), so it is already close to the cap.
FIX: Add a line under the rung on both monthly cards, e.g. "passes at the full edge only; at 3/4 of the edge the breach is 30.0% / 27.3% against the 15% cap". Either compute the edge margin for the two books or replace "not computed" with "above 0.75". Add one sentence to the RUNG caption next to the existing Aggressive / Growth Plus sentence.

## stale F2 [minor] report_texts.py line 1034 (rendered in the "חשוב לדעת" table, row "משקלי המוצרים נבחרו על ידך אחרי התוצאות"): "כל נקודות החוגה כבר נראו לפני ההקפאה."
PROBLEM: Not true for the chosen point. The grid seen before the freeze did not contain 40 / 30 / 30 for either TAA variant. The record says something different ("3 of 14 books seen before the freeze").
EVIDENCE: SPEC_FROZEN.md lines 28-37: seen weightings were 1/3 each, 50/25/25, 50/15/35, 50/50, 60/20/20, 20/40/40. SPEC line 5: "Not seen at the freeze: GR1 and GR2 in the main frame".
FIX: Replace with: "נקודת 40 / 30 / 30 לא הייתה ברשת שנראתה לפני ההקפאה; היא נבחרה אחרי התוצאות המלאות, מתוך מפת החוגה שנרשמה מראש."

## stale F3 [minor] Blend section: bullet "מול מזומן: ... ירידה היסטורית רדודה יותר, ב־0.7 … 1.4 pp. היתרון קיים, והוא קטן", column group "Same volatility: Growth diluted with BIL", caption "זו ההשוואה ההוגנת"; record line 68 ("a real but small advantage")
PROBLEM: The diluted comparators do not have the same volatility as the blends; they carry 1.3% to 4% more. That flatters the blend on Max DD and penalises it on CAGR. The page does not show the comparator's volatility. The drawdown advantage is one historical path, so "real" is stronger than the numbers allow.
EVIDENCE: study.json blends, vol blend vs diluted: 20%: 7.02 vs 7.31; 40%: 8.28 vs 8.45; 60%: 9.72 vs 9.94; 80%: 11.26 vs 11.41. Rescaling the diluted Max DD to the blend's volatility cuts the advantage from 1.07 / 1.21 / 1.35 / 0.66 pp to about 0.8 / 1.0 / 1.1 / 0.5 pp. At 20% Growth the blend is behind on both CAGR (−0.47 pp) and Sharpe (−0.011). The other stated ranges (CAGR −0.5 … +0.1, Sharpe −0.01 … +0.03, 3 of 4 interior points above both ends) are correct.
FIX: Either match the volatility exactly (solve the BIL share on the realised blend volatility) or add a Vol column for the diluted rows and write "approximately the same volatility (diluted rows 1-4% higher)". Soften to "ירידה היסטורית רדודה יותר בכ־0.5 עד 1.4 pp במסלול ההיסטורי האחד; בנקודת 20% הדילול במזומן מעט טוב יותר". Same edit in record line 68 and SPEC amendment O5.

## stale F4 [minor] "למה שני פודים" block, bullet "‏20% CAGR בתיק חודשי = מדרגת GROWTH PLUS": "המינוף לא נותן תוצאה טובה יותר מאשר יותר TAA 3x 1N, ולכן אין סיבה למנף"
PROBLEM: The verdict is stronger than the numbers. The levered 50/50 x1.25 row is equal or slightly better on most columns and worse only on the breach figure.
EVIDENCE: monthly.json ladder, x1.25 vs 65/35: CAGR 21.2% vs 20.8%, Excess Sharpe 1.148 vs 1.145, Max DD −17.5% vs −17.4%, GFC window −8.6% vs −9.9%, 2022 window −9.9% vs −10.7%, DD beyond −25% 10.5% vs 9.6%. Both pass GROWTH PLUS only.
FIX: "המינוף נותן תוצאה דומה (הבדלים בתוך הרעש), ודורש חשבון מרג׳ין; לכן אין סיבה למנף."

## stale F5 [minor] Card "אגרסיבי Aggressive", row "Rung passed": "holds down to 0.85 of the edge; AGGRESSIVE needs your approval"; RUNG table row label "אגרסיבי (AGGRESSIVE, awaiting approval)"
PROBLEM: Contradicts the same page, which says three times that no decision is needed while Aggressive is not offered ("כל עוד Aggressive לא מוצע, אין צורך להכריע עליה", and the TODO item "לא נדרשת").
EVIDENCE: study.json products.GR3.offered = false, step_over_product_below = 0.0117.
FIX: Change to "AGGRESSIVE rung: not decided, moot while not offered" in both places.

## stale F6 [minor] Defensive card "יותר תשואה": "‏60/40 עם 20% מתיק הצמיחה ו־20% cash"; and the "למה שני פודים" bullet "קפסולת המומנטום מחליפה את NDX-VXN בכל הקרן"
PROBLEM: "תיק הצמיחה" on the card is the four-pod monthly book of 1 October (the legend shows 7.7% TAA 3x 1N + 5.1% NDX-VXN), which the growth section says is "לא מוצעים יותר". "בכל הקרן" is therefore not true: a defensive option marked AT LAUNCH still holds NDX-VXN. Only the collapsed details block explains this.
EVIDENCE: Card legend 39.6 / 27.6 / 20 / 7.7 / 5.1 equals 20% of 38.4 / 25.6 / 18 / 18. Details row label: "פרוסת הצמיחה הישנה: התיק החודשי של ארבעה פודים".
FIX: Card text: "‏60/40 עם 20% מהתיק החודשי מ־1 באוקטובר (ארבעה פודים) ו־20% cash; כפי שפורסם". Bullet: "מחליפה את NDX-VXN במוצרי הצמיחה" (or add "חוץ מהשורה ההגנתית שפורסמה").

## stale F7 [minor] Summary bullet "מסחר חודשי בלבד; TAA 3x 1N כבר רץ בלייב"; Monthly card "TAA 3x 1N כבר רץ בלייב"; growth intro "TAA, המנוע הוותיק שכבר רץ בלייב"; record line 14 "TAA 3x 1N is wired live"
PROBLEM: The repository supports "wired" for both TAA variants, not "runs live" for both. The page uses "runs live" once for TAA 3x (reason for the Growth tilt) and once for TAA 3x 1N (reason Monthly can run first). If only one variant has a deployed pod, one of the two statements is overstated. I could not verify from this worktree which variant is deployed.
EVIDENCE: alpha/strategy_registry.py lines 62-63: both strategy_taa_df_btal_fallback_tqqq_vix_cash and ..._btal_1n_fallback_tqqq_vix_cash are MaturityTier.WIRED; CORE5 (line 104) is PM_READY. The page's own caveat row says only "מחווטים: TAA 3x, TAA 3x 1N".
FIX: Use "מחווט ללייב" for the variant that has no running pod, and name the variant that actually trades today.

## stale F8 [minor] All ten cards, row "Available": "CORE5 wired", "capsules wired + MR cost gate", "DV2-IND forward test"; ease table header "Pods: research / live" with Monthly "2 / 2" beside "Not wired: CORE5"
PROBLEM: "Available: CORE5 wired" reads as a status (CORE5 is wired), which is the opposite of the fact. "Pods ... live 2 / 2" beside "Not wired: CORE5" reads as both pods being live.
EVIDENCE: Registry: CORE5 PM_READY, not wired. Summary bullet: "חסר רק חיווט של CORE5".
FIX: Label "Available after" (or "Needs") with value "CORE5 wiring". Header "Pods: in research / in the live build".

## stale F9 [minor] VERIFICATION section, table "אחרי סוף החלון" and the "חשוב לדעת" row "השבועות אחרי סוף החלון"
PROBLEM: (a) Monthly Plus has no row; battery.json after_window.books has only GR1, GR2, GR3 and S9. (b) "2026 to date" does not chain with the calendar-year table: for all four books it is 0.1-0.3 pp below (1 + 2026*) × (1 + after window), with no note on the basis.
EVIDENCE: Growth: years 2026 = +19.78%, after window +1.05%, product = +21.04%, shown +20.8%. Growth Plus 26.72 vs 26.5; Aggressive 29.68 vs 29.4; Monthly 27.95 vs 27.9.
FIX: Add "old growth plus" to after_window in battery.py. State the basis of "2026 to date" (frame / reset) or compute it by chaining the same series.

## stale F10 [minor] "רגישות למשקלים" details: caption "סימון ״הטוב בשכונה״: מוצר שנמצא בין שלושת הטובים בשכונה שלו" and table column "Flags" = none for צמיחה, whose "DD beyond its limit" cell shows (#3)
PROBLEM: By the caption's definition Growth should carry the flag: it ranks 3rd of 19 on the breach figure. The code flags only Max DD and Excess Sharpe.
EVIDENCE: battery.py lines 213-217: `for key in ("dd", "xs")`. plateau.GR1 breach 2.385%, rank 3.
FIX: Either add "breach" to the flagged keys or write in the caption "(נבדק על Max DD ו־Excess Sharpe בלבד)".

## stale F11 [minor] Summary bullet "גודל": "התיק החודשי: $5M בפתיחה במודל הבית"; record line 18 "the Monthly book $5M"; growth menu table, row "החודשי פלוס מ־1 באוקטובר" capacity "–"
PROBLEM: The $5M applies to Monthly only; Monthly Plus is $2.5M at the open (stated correctly only in the capacity section). The old Monthly Plus reference row has no capacity because it is not in capacity.json.
EVIDENCE: capacity.json routes.MOO.recommended: S9 = 5,000,000; "old growth plus" (Monthly Plus) = 2,500,000.
FIX: "Monthly $5M ו־Monthly Plus $2.5M בפתיחה". Footnote the "–" as "not computed".

## stale F12 [minor] "למה שני פודים" bullet "מול התיק הישן של ארבעה פודים: ... תיקו לפי הכלל (Monthly גבוה ברוב המסלולים, אבל מתחת לרף). כלומר כמעט אותו תיק"
PROBLEM: The reused parenthesis is true for Sharpe only; on CAGR the old book is ahead on most paths.
EVIDENCE: versus.json "S9 incumbent launch | S13 old monthly": share_xs 0.681, share_cagr 0.315.
FIX: "(ב־Sharpe ה־Monthly החדש גבוה ברוב המסלולים, מתחת לרף; ב־CAGR הישן גבוה ב־68%)".

## stale F13 [minor] Readability, internal identifiers printed raw: table header "GR1 with one capsule replaced"; construction table row "GR1"; stand-in rows "GR1 with live NDX rule", "GR1 with ungated DV2 + HPI", "GR1 all wired today"; challenger rows "S13 Monthly of 2026-10-01", "S0 …", "S8 GR1 75 / DEF 25", "S9 Monthly", "S10 old G3"; decay table rows "S4 core + satellites", "S11 cluster parity", "S1 no momentum"; monthly reference rows "old monthly (2026-10-01)", "old monthly plus (2026-10-01)"; note "‏S9 ו־S13: …" and legend "שמות בטבלאות: GR1 = Growth … S9 = Monthly"
PROBLEM: The legend line helps, but the codes still force the reader to translate, and "old G3" and "DEF" are not in the legend. The keys "S9 incumbent launch", "old growth plus" and "nb GR1 …" do not appear on the page (the minimax rival is correctly rendered as "TAA 3x 30 / 35 / 35").
EVIDENCE: grep of the rendered HTML: 7 distinct GR1 strings, S-codes in two tables.
FIX: Map at render time: GR1 → Growth, "old monthly (2026-10-01)" → "Monthly of 2026-10-01 (four pods)", "S8 GR1 75 / DEF 25" → "Growth 75 / Defensive 25". Drop the S-prefixes or keep them only in the challenger table with the legend directly under it.

## stale F14 [minor] Readability, list items over 300 characters (14): TODO "כלל התפריט" (452); "‏20% CAGR בתיק חודשי" (400); "למה בלי מומנטום" (374); summary "‏Growth מול Monthly" (371); summary "שני תיקים חודשיים" (356); summary "‏Aggressive" (355); TODO "הוחלט" (345); verification "המחקר הקודם" (326); summary "סולם אחד" (325); summary "גודל" (313); "חישוב של הסוקר" (309); margin "מול Growth Plus" (306); TODO "פתוח: שער העלות" (304); summary "אז למה Growth נשאר היעד" (303)
PROBLEM: These are still dense walls, mostly in the "מה החלטתי" summary, which is the part read first. The "כלל התפריט" sentence is repeated verbatim four times (summary, growth intro, growth notes, TODO).
EVIDENCE: Character counts from the rendered <li> elements.
FIX: In the summary keep one line of numbers per product and move the paired-path shares to the versus section. Split "למה בלי מומנטום" and "20% CAGR" into two bullets each. State the menu rule once and refer to it elsewhere ("Aggressive לא מוצע, ראה כלל התפריט").

## stale F15 [minor] Defensive menu table, name cell "60/40 בלי cash (לא עומד בכלל)"
PROBLEM: The only Hebrew cell or list item on the page that starts with a Latin/digit token without an RLM; the leading "60/40" can jump to the wrong side in RTL. All other such items carry U+200F.
EVIDENCE: Scan of 3,688 p/li/td/th/summary/heading elements: 1 hit.
FIX: Prefix with the same R (U+200F) helper used elsewhere, or reword to start in Hebrew ("בלי cash: 60/40 …").

## stale F16 [minor] report_body.html CSS: `.card > .code` (mono 12px, no wrap) and `.prod5 { grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)) }`
PROBLEM: There is no horizontal page scroll at 400px. At desktop and at 440-480px widths, where cards are 195-215px wide, content spills outside the card border, worst on the Monthly Plus card (the file name "fund_growth_monthly_plus.yaml"). Cosmetic.
EVIDENCE: Measured in the browser with all details open: body width 400px gives 0 elements outside 0-400 and all cards 360px. Overflow beyond the card: at 1120px Monthly Plus 16px, Aggressive 1px; at 440px Monthly Plus 29px, Aggressive 14px, יעד אידיאלי 7px. No page-level overflow at any tested width.
FIX: `.card > .code { overflow-wrap: anywhere; }` and `.card > * { min-width: 0; }`, or raise the minmax to 215px.

## stale F17 [minor] report_body.html script, chart tooltip: `let x = e.clientX - hr.left + 12; if (x + 230 > hr.width) x -= 250;`
PROBLEM: On a phone-width chart (host about 334px wide) the flip gives a negative left for any pointer x between about 104 and 250, so the tooltip is cut off past the chart's left edge. From the code only; not pointer-tested.
EVIDENCE: host width ≈ 400 − 40 − 26 = 334; for x = 150, left = −100px.
FIX: Clamp: `x = Math.max(0, Math.min(x, hr.width - tip.offsetWidth))`.

## stale F18 [minor] "כמה להאמין" table header: `<th>P(Excess Sharpe < 1.0 / 0.75 / 0.5)</th>`
PROBLEM: Raw unescaped "<" in the HTML. Browsers render it, but tag-stripping tools and strict parsers swallow the header and the next one.
EVIDENCE: Raw HTML of fund_products_v5.html; my own tag-stripper lost this header.
FIX: Emit "&lt;" (or write "below").
