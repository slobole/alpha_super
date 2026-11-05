"""Build the visual-first HTML page (charts embedded) for the Beyond 60/40 study."""
import base64
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
C = HERE / "out" / "charts"


def img(n):
    return "data:image/png;base64," + base64.b64encode((C / n).read_bytes()).decode()


v = pd.read_csv(HERE / "out" / "vault" / "vault_results.csv", index_col=0)
rows = "".join(
    f"<tr><td>{i}</td><td>{r.cagr:.1%}</td><td>{r.sharpe:.2f}</td><td>{r.max_dd:.1%}</td><td>{r.y2023:.1%}</td>"
    f"<td>{r.y2024:.1%}</td><td>{r.y2025:.1%}</td><td>{r.y2026:.1%}</td></tr>"
    for i, r in v.loc[["T1 (chosen)", "B0 (author rule)", "60/40", "EW3"]].iterrows())

html = f"""<title>Beyond 60/40 Robustness</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column, charts full width, numbers in mono */
:root {{ --bg:#fbfaf7; --fg:#1d2430; --muted:#6b7380; --line:#e3e0d8; --navy:#1f3a5f; --gold:#b8860b; --red:#c0392b; --good:#2e7d32; --card:#ffffff;
  --font-body:"Heebo",system-ui,sans-serif; --font-num:"IBM Plex Mono",ui-monospace,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#15191f; --fg:#e8e6e0; --muted:#9aa3ad; --line:#2d333c; --navy:#8fb3e0; --gold:#e0b84a; --red:#e8705f; --good:#7cc47f; --card:#1d232b; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#15191f; --fg:#e8e6e0; --muted:#9aa3ad; --line:#2d333c; --navy:#8fb3e0; --gold:#e0b84a; --red:#e8705f; --good:#7cc47f; --card:#1d232b; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--font-body); font-size:16px; line-height:1.55; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); padding-block:24px 48px; }}
main {{ max-width:1040px; margin:0 auto; direction:rtl; }}
h1 {{ font-size:1.9rem; margin:0 0 .2rem; text-wrap:balance }} h2 {{ font-size:1.25rem; margin:2.2rem 0 .6rem; border-top:1px solid var(--line); padding-top:1rem }}
.sub {{ color:var(--muted); margin:0 0 1.2rem }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:1rem 0 }}
.tile {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; direction:ltr; text-align:left; min-width:0 }}
.tile .k {{ font-size:.78rem; color:var(--muted); letter-spacing:.03em; text-transform:uppercase }} .tile .v {{ font-family:var(--font-num); font-size:1.5rem; font-weight:500; margin-top:2px }} .tile .w {{ font-size:.8rem; color:var(--muted) }}
figure {{ margin:1rem 0 }} figure img {{ width:100%; max-width:100%; display:block; border:1px solid var(--line); background:#fff }} figcaption {{ font-size:.9rem; color:var(--muted); margin-top:.4rem }}
p {{ max-width:70ch }} p.en {{ direction:ltr; text-align:left; font-family:var(--font-num); font-size:.85rem; color:var(--muted); max-width:none }}
table {{ border-collapse:collapse; width:100%; direction:ltr; font-family:var(--font-num); font-size:.85rem; font-variant-numeric:tabular-nums }} .tw {{ overflow-x:auto; min-width:0 }}
th,td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }} th:first-child,td:first-child {{ text-align:left }} th {{ color:var(--muted); font-weight:500 }}
td.best {{ color:var(--navy); font-weight:500 }} .pass {{ color:var(--good) }} .fail {{ color:var(--red) }}
ul {{ padding-inline-start:1.2rem; max-width:70ch }} li {{ margin:.3rem 0 }}
</style>
<main>
<h1>Beyond 60/40: מחקר רובוסטיות והחלטה</h1>
<p class="sub">מניות, אג"ח ארוך וזהב במשקל הפוך לתנודתיות, יעד תנודתיות 8%. נבדק על 1970–2022; כספת 2023–2026 נפתחה פעם אחת. כל המספרים מעל ריבית מזומן (Fed funds), עלות 5bp.</p>

<div class="tiles">
<div class="tile"><div class="k">Sharpe over cash 1970–2022</div><div class="v">0.60–0.65</div><div class="w">T1 · 60/40 0.45 · B0 0.43</div></div>
<div class="tile"><div class="k">Max drawdown</div><div class="v">−11%</div><div class="w">−10.7% to −11.6% · 60/40 −36.3%</div></div>
<div class="tile"><div class="k">With cash at 0%</div><div class="v">0.30</div><div class="w">the edge needs idle cash in T-bills</div></div>
<div class="tile"><div class="k">Beats 60/40 every decade</div><div class="v">No</div><div class="w">trails in the 1990s and 2010s bulls</div></div>
<div class="tile"><div class="k">Vault 2023–26</div><div class="v">42nd pct</div><div class="w">inside a wide band</div></div>
</div>

<p><b>השורה התחתונה (אחרי ביקורת בלתי תלויה).</b> הכלל המקורי (B0) הוא מכונת ירידות, לא מכונת Sharpe: מעל מזומן הוא שקול ל־60/40. עם מסנן הטרנד של הכותב (T1: נכס מתחת לממוצע 10 חודשים יוצא למזומן) הוא מנצח את 60/40 מעל מזומן, 0.60–0.65 מול 0.45, עם ירידה מקסימלית של כ־11% מול 36%. שני סייגים מהותיים: היתרון תלוי בכך שהמזומן הפנוי (כשליש מהתיק בממוצע) מרוויח ריבית, וב־0% הוא יורד ל־0.30, מתחת ל־60/40; והוא לא עבר את הרף הקשוח שקבענו מראש, לנצח את 60/40 כמעט בכל עשור: הוא מפסיד לו בשווקי השור של שנות ה־90 וה־2010. זו אסטרטגיה דפנסיבית טובה, לא תחליף ל־60/40 בכל תקופה. בחירת המסנן נעשתה מתוך גריד שנוסף אחרי התוצאות הראשונות.</p>

<figure><img src="{img('equity_drawdown.png')}" alt="Growth of one dollar and drawdowns 1970-2026"><figcaption>$1 מ־1970, סקאלה לוגריתמית. האפור מימין: הכספת 2023+. T1 (כחול) לא ירד יותר מ־11% במוסכמת המחקר (11.6% בביצוע ריאלי, יום אחד מאוחר יותר).</figcaption></figure>
<figure><img src="{img('decade_sharpe.png')}" alt="Sharpe over cash by decade"><figcaption>Sharpe מעל מזומן לפי עשור. T1 מפסיד ל־60/40 רק בשני עשורי השור הארוכים (1990s, 2010s) ומנצח בכל השאר, כולל כל עשורי האינפלציה.</figcaption></figure>

<h2>למה הטרנד עובד כאן</h2>
<figure><img src="{img('regimes.png')}" alt="Return over cash by regime"><figcaption>תשואה שנתית מעל מזומן לפי משטר. כשמניות ואג"ח נעות יחד והמניות יורדות, B0 מפסיד 3% בשנה ו־T1 מרוויח כ־1%. במשטרים הטובים ההבדל קטן.</figcaption></figure>
<figure><img src="{img('tilt_grid.png')}" alt="Trend tilt plateau"><figcaption>המישור של הטרנד: אורך ממוצע 6–12 חודשים × עוצמה k. k=0.25 נותן חצי משקל מתחת לממוצע, k=0.5 (T1) יוצא מהנכס לגמרי, ו־k=0.75 ו־1 כבר מוסיפים שורט. כל 8 התאים בלי שורט עוברים את כלל האימוץ; השורט משתווה ב־Sharpe אבל מעמיק ירידות. הגריד הזה נוסף אחרי התוצאות הראשונות.</figcaption></figure>
<figure><img src="{img('plateau.png')}" alt="Parameter plateau"><figcaption>חלון תנודתיות × יעד. עם הטרנד כל 60 התאים מנצחים את 60/40, וגם 108 תאים בגריד רחב יותר. יעד התנודתיות כמעט לא משנה את ה־Sharpe, רק את עומק הירידה.</figcaption></figure>

<h2>הכספת</h2>
<figure><img src="{img('vault.png')}" alt="Vault 2023-2026"><figcaption>2023–2026, נפתח אחרי הקפאת העיצוב (ופעם נוספת אחרי תיקון באג במנוע). שוק שור בכל שלושת הנכסים: B0 עשה 11.2% בשנה, T1 7.9%, 60/40 12.0%. זו עלות הביטוח. ה־12.4% המצטברים מעל מזומן של T1 יושבים באחוזון 42; הטווח רחב, אז זו בדיקה חלשה.</figcaption></figure>
<div class="tw"><table><tr><th>2023–2026-10</th><th>CAGR</th><th>Sharpe over cash</th><th>Max DD</th><th>2023</th><th>2024</th><th>2025</th><th>2026 YTD</th></tr>{rows}</table></div>

<h2>מה לא שינה את התמונה</h2>
<div class="tw"><table><tr><th>Change</th><th>Sharpe vs B0</th><th>Verdict</th></tr>
<tr><td>World stocks 50/50 (1991→)</td><td>0.64 vs 0.71</td><td class="fail">worse, not noise</td></tr>
<tr><td>10y bond instead of 25y</td><td>0.42 vs 0.43</td><td>tie, weaker 2022</td></tr>
<tr><td>Gold + commodities (DBC)</td><td>0.68 vs 0.79</td><td class="fail">better 2022 only</td></tr>
<tr><td>Equal / ERC / MinVar weights</td><td>0.52 / 0.37 / 0.34 vs 0.43</td><td>equal wins only via the 1970s; MinVar concentrates</td></tr>
<tr><td>21 / 126 / 252-day vol, EWMA</td><td>≤ +0.02</td><td>some pass the rule; noise, not adopted</td></tr>
<tr><td>Leverage 1.5× (financed)</td><td>0.39 vs 0.43</td><td class="fail">Sharpe falls</td></tr>
<tr><td>Leverage 1.25–2× capped, on T1</td><td>0.65–0.67 vs 0.65</td><td>passes the rule; +0.4–1.0% CAGR, DD −13%: a dial</td></tr>
<tr><td>VIX hedge</td><td>not tested here</td><td>PAKAL, on the author's levered trend book: removes ~11% of worst-day damage; 10% VIXY cut return 7.3% → 1.9%</td></tr>
</table></div>

<h2>משקיע בשקלים</h2>
<p>(נמדד על B0, לא על T1.) בלי גידור, ה־Sharpe נחתך לחצי (0.71 → 0.33) והירידות הגרועות ביותר הן של השקל, לא של האסטרטגיה. גידור מלא משחזר את המסלול הדולרי; גידור 50% נותן את הירידה הרדודה ביותר. הגידור הפעיל של הכותב לא עובד על דולר-שקל.</p>
<div class="tw"><table><tr><th>1993–2022, ILS investor</th><th>CAGR</th><th>Sharpe over IL cash</th><th>Max DD</th></tr>
<tr><td>USD investor (reference)</td><td>7.6%</td><td>0.71</td><td>−16.2%</td></tr>
<tr><td>Unhedged</td><td>8.4%</td><td class="fail">0.33</td><td>−17.5%</td></tr>
<tr><td>Full hedge</td><td>10.7%</td><td class="best">0.70</td><td>−17.0%</td></tr>
<tr><td>50% hedge</td><td>9.6%</td><td>0.54</td><td class="best">−12.6%</td></tr>
<tr><td>Active trend hedge (author)</td><td>6.4%</td><td class="fail">0.34</td><td>−18.6%</td></tr>
</table></div>

<h2>מקום בספר</h2>
<p>T1 <b>אינו</b> אותו מנוע כמו CORE5 (גרסה קודמת של העמוד טענה זאת). CORE5 הוא חמישה סליבים קבועים של 20% (SPY, IEF, GLD, DBC, UUP) עם מתג טרנד אדפטיבי, בלי משקל לפי תנודתיות ובלי יעד תנודתיות; הקורלציה החודשית 0.61. לכן כלל "פוד אחד לכל מנוע" לא מוציא אותו. על 2008–2022: T1 מול CORE5, Sharpe 0.79 מול 0.82 (ומ־2012 0.57 מול 0.94). הוספת 20% T1 לספר הדפנסיבי: 1.01 מול 1.00, ומ־2012 1.00 מול 1.05. המקרה לספר חלש, וההחלטה שלך.</p>
<figure><img src="{img('weights_t1.png')}" alt="T1 allocation over time"><figcaption>ההקצאה של T1 לאורך הזמן. כשנכס מתחת לממוצע שלו, כל המשקל שלו עובר למזומן.</figcaption></figure>

<h2>הכלל שנבחר</h2>
<ul>
<li>סוף חודש: משקל כל נכס ∝ 1/תנודתיות 63 יום. נכסים: VTI, TLT, GLD.</li>
<li>אותו יום: נכס שמחירו מתחת לממוצע 10 חודשים יוצא כולו למזומן; מעל הממוצע הוא מקבל את מלוא משקלו.</li>
<li>יומי: אם תנודתיות 63 יום של הסל הבסיסי מעל 8.5%, חשיפה = 8% ÷ תנודתיות. בלי מינוף, בלי שורט. היתרה ב־BIL, וזה תנאי ליתרון.</li>
</ul>
<p class="en">Record: docs/research/BEYOND_6040_20261105.md · spec, tables and scripts: scripts/research/beyond_6040_20261105 · strategy: strategies/taa_beyond_6040/strategy_taa_beyond_6040_trend_tilt.py (PM_READY, no book slot; live route on branch t1-wiring-20261105, not merged). Audited 2026-11-05: corrections listed in the record. Sharpe over Fed funds unless marked raw; the author's published Sharpe is raw.</p>
</main>
"""
out = HERE / "out" / "beyond_6040_page.html"
out.write_text(html, encoding="utf-8")
print(out, len(html) // 1024, "KB")
