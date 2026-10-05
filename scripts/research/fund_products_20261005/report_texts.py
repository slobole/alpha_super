"""Texts and tables of the fund-products report v5: statistics and table labels in English, short Hebrew explanations.

Rules kept from v4: metric names in English; a Hebrew line never starts with a Latin token (RLM before it); numbers
inside Hebrew text sit in an LTR span.
"""

from __future__ import annotations

import html as _html
from pathlib import Path
import re as _re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fund_products_20260930"))
import v4_texts as V4  # noqa: E402  (names, chips and reasons of the defensive rows, kept as published)

R = "‏"   # RLM


def L(s: str) -> str:
    return f'<span class="ltr">{s}</span>'


def rl(x: str) -> str:
    """A Hebrew line must not start with a Latin token: put an RLM before a line that opens with one (also inside <b>)."""
    return R + x if _re.match(r'\s*(<b>)?\s*(<span class="ltr">|[A-Za-z0-9×+−])', x) else x


GR = {"GR1": ("Growth", "צמיחה", "var(--s3)"), "GR2": ("Growth Plus", "צמיחה פלוס", "var(--s5)"), "GR3": ("Aggressive", "אגרסיבי", "var(--s4)")}
S9 = "S9 incumbent launch"           # internal key (not renamed): the Monthly book, TAA 3x 1N 60 / CORE5 40 (owner decision 2026-10-05)
M2 = "old growth plus"               # internal key of the dropped 65 / 35 book: still in the JSON files, never shown as a product
AGG_APPROVED = "2026-10-05"          # the owner approved the AGGRESSIVE rung (max DD >= -27%, 15% cap at -30%) on this date
LEV_KEY = "fixed_140"                # study.json margin["GR1 -> GR3"][LEV_KEY]: Growth at a fixed 1.40x, an alternative route, not a product
S13 = "S13 old monthly (2026-10-01)"   # the four-pod monthly book of 2026-10-01, a reference row
OLDP = "old monthly plus (2026-10-01)"
S0 = "S0 equal capital (registered default)"
GR1_L = "T2 MR -> BIL (GR1-L)"
HARD = {"GROWTH": "p20", "GROWTH PLUS": "p25", "AGGRESSIVE": "p30"}
CRISIS_LABEL = {"gfc": "2008 GFC", "q4_2018": "Q4 2018", "covid": "COVID 2020", "bear_2022": "2022 bear", "tariffs_2025": "Tariffs 2025"}
BOOK_EN = {"GR1": "Growth", "GR2": "Growth Plus", "GR3": "Aggressive", S9: "Monthly", M2: "Monthly Plus",
           S13: "Monthly of 2026-10-01 (four pods)", OLDP: "Monthly, more return, of 2026-10-01 (four pods)",
           GR1_L: "Growth, MR slot in BIL", "capsule TAA 3x": "TAA 3x alone", "capsule TAA 3x 1N": "TAA 3x 1N alone", "capsule MOM": "Momentum capsule alone",
           "capsule MR": "MR capsule alone", "capsule DEF": "CORE5 60 / BTAL_QQQ 40", "S10 old G3": "Old G3 (TAA 3x 50 / NDX-VXN 50)"}
# Readable names for internal book ids: no raw code (GR1, S0, S8, S13, DEF, "old G3") is printed in a table.
SHOW = {S9: "Monthly (TAA 3x 1N 60 / CORE5 40)", S13: "Monthly of 2026-10-01 (four pods)", M2: "Monthly Plus", OLDP: "Monthly, more return, of 2026-10-01 (four pods)",
        "GR1": "Growth (TAA 3x 40 / 30 / 30)", S0: "Equal capital, 1/3 each (registered default)", "S1 no momentum": "No momentum (TAA 3x 50 / MR 50)",
        "S2 no MR": "No MR (TAA 3x 50 / momentum 50)", "S3 no TAA": "No TAA (momentum 50 / MR 50)", "S4 core + satellites": "TAA 3x 50 / 25 / 25",
        "S5 MR tilt": "MR tilt (TAA 3x 50 / momentum 15 / MR 35)", "S6 equal pods": "Equal pods (TAA 3x 20 / 40 / 40)", "S7 inverse vol (walk-forward)": "Inverse volatility, walk-forward",
        "S8 GR1 75 / DEF 25": "Growth 75 / Defensive core 25", "S10 old G3": "TAA 3x 50 / NDX-VXN 50 (the old growth book)", "S11 cluster parity": "Cluster parity (TAA 3x 25 / 25 / 50)",
        "S12 inverse vol (fixed)": "Inverse volatility, fixed (31 / 33 / 36)", "old monthly (2026-10-01)": "Monthly of 2026-10-01 (four pods)",
        "old monthly plus (2026-10-01)": "Monthly, more return, of 2026-10-01 (four pods)"}


def show(n: str) -> str:
    return SHOW.get(n, n).replace("GR1", "Growth")
MONTHLY_HE = {S9: "החודשי", M2: "החודשי פלוס"}


def build(S: dict, B: dict, C: dict, X: dict, D: dict, P: dict, A6: dict, h: dict, V: dict, M: dict | None = None) -> dict:
    pct, sg, f2, usd, mix, td, table, tr, name_cell, lt, yes, heat = (h[k] for k in ("pct", "sg", "f2", "usd", "mix", "td", "table", "tr", "name_cell", "lt", "yes", "heat"))
    pie = h["pie"]
    M = M or {}
    K = S["books"]
    HL = B["edge_decay"]["headline"]
    out: dict = {}
    fin = {n: S["products"][n]["final"] for n in GR}
    en = lambda n: BOOK_EN.get(n, n)  # noqa: E731
    cap_of = lambda n, route="worked+blocks": (C.get("books", {}).get(n, {}).get("routes", {}).get(route, {}) or {})  # noqa: E731

    def capstr(n: str, route: str = "worked+blocks") -> str:
        r = cap_of(n, route)
        wall = C.get("books", {}).get(n, {}).get("btal_wall")
        if r and route == "worked+blocks" and wall and r.get("recommended") and wall < r["recommended"]:
            return usd(wall)                              # the BTAL ownership wall binds before the route figure
        return "–" if not r else usd(r.get("recommended"), r.get("at_grid_top", False)) if r.get("recommended") else "&lt;$0.5M"

    g1, g2, g3, s9, m2 = K["GR1"], K["GR2"], K["GR3"], K[S9], K[M2]
    ED_ = B["edge_decay"]["scenarios"]
    # Conservative case: every engine keeps 3/4 of its historical excess return, at MODEL costs. Stress case: half the
    # excess return and +5 bps per side. Either may be missing for a book the battery did not carry (shown as a dash).
    cons = lambda n: ED_["all at 0.75"].get(n) or {}  # noqa: E731
    stress = lambda n: (HL.get(n) or {}).get("floor") or {}  # noqa: E731
    dead = lambda n: (ED_["TAA dead"].get(n) or {}).get("xs")  # noqa: E731
    dep = B["dependence"]["products"]

    def tiles(items: list) -> str:
        return '<div class="tiles">' + "".join(f'<div class="tile"><span>{k}</span><b>{v}</b></div>' for k, v in items) + "</div>"

    def kv(rows: list) -> str:
        """Label / value rows; a row with value None is a full-width note."""
        return '<div class="kv">' + "".join(f'<div class="kvn">{k}</div>' if v is None else f'<div class="r"><span>{k}</span><b>{v}</b></div>' for k, v in rows if k) + "</div>"

    def notes(items: list) -> str:
        """A caption as a full-width bullet list, one idea per bullet."""
        return '<ul class="notes">' + "".join(f"<li>{rl(x)}</li>" for x in items if x) + "</ul>"

    def points(items: list) -> str:
        """An intro as a full-width bullet list in the body size."""
        return '<ul class="plain">' + "".join(f"<li>{rl(x)}</li>" for x in items if x) + "</ul>"

    def bar(share: float) -> str:
        """Where a paired share stands against the pre-registered 80% bar."""
        return "מעל הרף של" if share >= 0.82 else "על הרף של" if share >= 0.79 else "מתחת לרף של"

    def who(share: float, a: str, b: str) -> str:
        """Plain reading of a paired share of book a over book b."""
        if share >= 0.79:
            return f"{L(a)} מוביל בבירור"
        if share >= 0.60:
            return f"תיקו לפי הכלל ({L(a)} גבוה ברוב המסלולים, אבל מתחת לרף)"
        if share > 0.40:
            return "תיקו"
        if share > 0.21:
            return f"תיקו לפי הכלל ({L(b)} גבוה ברוב המסלולים, אבל מתחת לרף)"
        return f"{L(b)} מוביל בבירור"

    # ── header ──
    out["EYEBROW"] = ("Fund products · v5.2 · 2026-10-05 · frozen plan 70f5c22 · owner decisions of 2026-10-05: Growth 40 / 30 / 30, "
                      "Growth Plus the same with TAA 3x 1N, two-pod monthly books, the backtest as the headline · LONG 2008-03 → 2026-08, gross, fair cash unless stated")
    out["LEDE"] = (f"ההגנתי נבדק ונשאר. הצמיחה נבנתה מחדש משלוש קפסולות: {L('40%')} {L('TAA')}, {L('30%')} מומנטום, {L('30%')} {L('MR')}. לידה שני תיקים חודשיים של שני פודים, שיכולים לרוץ ראשונים. "
                   f"מספרי הכותרת הם ה־{L('backtest')}; בפרק ״כמה להאמין״ יש תרחיש שמרני אחד ותרחיש לחץ אחד.")

    # ── summary ──
    t2, t1 = S["slots"][GR1_L]["frames"], S["slots"]["T1 MOM -> BIL"]["frames"]
    vs = V[f"GR1 | {S9}"]
    vf, vb = vs["frames"], vs["blocks"]
    rev = next(c for c in S["challenges_reverse"] if c["default"] == S9)
    gate = S["mr_gate_breakeven"]
    nb_name = lambda x: ("TAA 3x " + " / ".join(f"{float(v) * 100:.0f}" for v in x.split()[-1].split("/"))) if x.startswith("nb GR1") else show(x)  # noqa: E731
    # Growth against the Monthly book: every verdict word is computed from versus.json.
    by_cost = (f"{L(pct(vf['main']['share_xs'], 0))} מהמסלולים בעלויות המודל ({bar(vf['main']['share_xs'])} {L('80%')}), "
               f"{L(pct(vf['plus5']['share_xs'], 0))} ב־{L('+5 bps')} ו־{L(pct(vf['plus10']['share_xs'], 0))} ב־{L('+10 bps')}")
    by_block = (f"{L(pct(vb['A']['main']['share_xs'], 0))} ב־2008 עד 2012, {L(pct(vb['B']['main']['share_xs'], 0))} ב־2012 עד 2021, "
                f"{L(pct(vb['C']['main']['share_xs'], 0))} מ־2022 ו־{L(pct(vb['RECENT']['main']['share_xs'], 0))} בשלוש השנים האחרונות")
    since22 = who(vb["C"]["main"]["share_xs"], "Growth", "Monthly")
    at5 = f"מול התיק החודשי, {L('Growth')} גבוה ב־{L('Sharpe')} ב־{by_cost}"
    d_g1, d_s9 = dead("GR1"), dead(S9)
    p3 = S["products"]["GR3"]
    g3_off, g3_step = bool(p3.get("offered", True)), p3.get("step_over_product_below")
    step_txt = f"{(g3_step or 0) * 100:.1f} pp"
    v21 = (V.get("GR2 | GR1") or {}).get("frames", {}).get("main")
    not_offered = (f"מוצר גבוה יותר מוצע רק אם הוא מוסיף לפחות {L('1.5 pp')} {L('CAGR')} (נקבע מראש). הצעד מ־{L('Growth Plus')} ל־{L('Aggressive')} הוא רק {L(step_txt)}, ולכן {L('Aggressive')} לא מוצע. "
                   f"התפריט: {L('Growth')}, {L('Growth Plus')} ושני התיקים החודשיים. {R}{L('Aggressive')} נשאר בעמוד כשורת השוואה.")
    not_offered_ref = "לא מוצע (ראה כלל התפריט בפרק הצמיחה)."
    rung3 = p3.get("strictest_rung")                     # the strictest rung Aggressive passes (None = no rung)
    hk3 = HARD.get(rung3 or "", "p30")                   # its limit: -30% for AGGRESSIVE, -25% for GROWTH PLUS
    needs_agg = rung3 == "AGGRESSIVE"                    # it passes only the new rung the owner has not formally confirmed
    fk = lambda k_: "none" if k_ is None else f"{k_:.2f}"  # noqa: E731
    em3 = (((B["edge_decay"].get("edge_margin", {}).get(fin["GR3"]) or {}).get("by_limit") or {}).get(hk3) or {}).get("lowest_passing_k")
    em2 = (((B["edge_decay"].get("edge_margin", {}).get(fin["GR2"]) or {}).get("by_limit") or {}).get("p25") or {}).get("lowest_passing_k")
    EM = B["edge_decay"].get("edge_margin", {})
    hk_of = lambda n: HARD.get(K[n].get("strictest_rung") or "", "p20")  # noqa: E731

    def margin_k(n: str):
        """Lowest share of the edge at which the book still passes its rung's breach cap (None when the battery has no curve)."""
        e = EM.get(n)
        if not e:
            return None
        return ((e.get("by_limit") or {}).get(hk_of(n)) or {}).get("lowest_passing_k", e.get("lowest_passing_k"))

    mk_s9 = margin_k(S9)
    cons_breach = lambda n: cons(n).get(hk_of(n))  # noqa: E731
    mw = s9["weights"]
    m_short = f"{mw.get('taa3x_1n', 0) * 100:.0f} / {mw.get('core5', 0) * 100:.0f}"
    m_mix = f"TAA 3x 1N {mw.get('taa3x_1n', 0) * 100:.0f} / CORE5 {mw.get('core5', 0) * 100:.0f}"
    rung_m, hk_m = s9.get("strictest_rung"), hk_of(S9)
    lim_m = "−" + hk_m[1:] + "%"
    thin_m = mk_s9 is None or mk_s9 > 0.80                     # thinner than every capsule product
    monthly_margin = ((f"המדרגה מחזיקה עד {L(f2(mk_s9))} מהיתרון" if mk_s9 is not None else "מרווח המדרגה לא חושב")
                      + f"; בתרחיש השמרני {L(pct(cons_breach(S9)))} מהמסלולים מעבר ל־{L(lim_m)}, מול תקרה של {L('15%')}")
    more_risk_m = rung_m != "GROWTH"                            # the monthly book sits on a higher rung than Growth
    agg_ok_txt = f"המדרגה {L('AGGRESSIVE')} ({L('Max DD ≥ −27%, DD beyond −30% ≤ 15%')}) אושרה על ידך ב־5 באוקטובר 2026."
    LEV = (S["margin"].get("GR1 -> GR3") or {}).get(LEV_KEY) or {}
    cg1, cs9 = cons("GR1"), cons(S9)
    mcap = C.get("books", {}).get(S9, {}).get("ease", {})
    not_wired_m = ", ".join(h["POD"].get(a, a) for a in mcap.get("not_wired", [])) or "none"
    line3 = lambda b: L(f"CAGR {pct(b['q']['cagr'])} · Excess Sharpe {f2(b['q']['xs'])} · Max DD {sg(b['q']['dd'])}")  # noqa: E731
    out["SUMMARY"] = "".join(f"<li>{rl(x)}</li>" for x in [
        "<b>הגנתי:</b> נשאר כמו שפורסם. בדקתי, ואין מה לעדכן.",
        f"<b>שלושה מנועים שונים:</b> {L('TAA')} (הקצאה טקטית), מומנטום במניות {L('Nasdaq-100')}, והיפוך לממוצע במניות {L('S&amp;P 500')}. אף אחד לא יודע איזה מנוע יחזיק מעמד, ולכן לא נשענים על אחד.",
        f"<b>משקלים בלי אופטימיזציה:</b> {L('40 / 30 / 30')}. הטיה קלה ל־{L('TAA')}, המנוע הוותיק שכבר רץ בלייב. ברירת המחדל הייתה הון שווה, וההבדל ביניהם בתוך הרעש.",
        f"<b>יותר תשואה:</b> אותו תיק עם {L('TAA 3x 1N')} ({L('Growth Plus')}). זה יותר {L('TQQQ')}, לא עוד אלפא.",
        (f"<b>{L('Aggressive')}:</b> {L('60%')} ב־{L('TAA 3x 1N')}, ושני הלוויינים {L('20%')} כל אחד. הכי הרבה תשואה בסולם, והכי תלוי במנוע אחד: ה־{L('TAA')} נושא {L(pct(dep['GR3']['risk_share']['TAA'], 0))} מהסיכון."
         if g3_off else f"<b>{L('Aggressive')} ירד מהתפריט:</b> הוא מוסיף מעט מדי מעל {L('Growth Plus')}, לפי כלל שנקבע מראש."),
        f"<b>מינוף כחלופה:</b> מינוף על {L('Growth')} הוא דרך אחרת להגיע לרמת התשואה של {L('Aggressive')}, בלי לרכז את הסיכון במנוע אחד. רק בשלב הקרן, עם חשבון מרג׳ין אחד; זו חלופה, לא מוצר בתפריט.",
        f"<b>תיק חודשי אחד, של שני פודים</b> ({L(m_mix)}): פשוט, מסחר חודשי בלבד, ניתן להגדלה, ויכול לרוץ ראשון."
        + (f" הוא על מדרגת {L(rung_m or 'none')}: יותר סיכון מ־{L('Growth')}, לא פחות." if more_risk_m else ""),
        f"<b>למה בלי מומנטום ובלי {L('BTAL_QQQ')} בחודשי:</b> המומנטום לא שיפר שם את ה־{L('Sharpe')}; {L('BTAL_QQQ')} הוא כמעט אותו מנוע כמו {L('TAA 3x 1N')}; ו־{L('CORE5')} נחוץ כרגל הגנתית.",
        f"<b>למה {L('Growth')} נשאר היעד:</b> לתיק החודשי יש מנוע תשואה אחד. " + f"אם ה־{L('TAA')} מפסיק לעבוד אין לו גיבוי" + (", והמרווח שלו במדרגת הסיכון דק." if thin_m else "."),
        f"<b>מספר הכותרת הוא ה־{L('backtest')}:</b> העמלות וההחלקה כבר בפנים. מה שלא בפנים: שום דבר כאן אינו מחוץ למדגם, והשנים האחרונות חלשות מהעשור שלפניהן.",
        f"<b>התאמת סיכון ללקוח:</b> חוגה אחת בין ההגנתי ל־{L('Growth')}, לא מוצר חדש לכל לקוח.",
        f"<b>מה עוד פתוח:</b> חיווט של {L('CORE5')}; מדידת עלות הביצוע של ה־{L('MR')} בנייר; וגודל קטן למוצרי הקפסולות עד שיש מסלול ביצוע אחר.",
    ])

    # ── DEFENSIVE (carried from the 2026-10-01 study) ──
    AR = A6["rows"]
    out["DEF_INTRO"] = (
        f"<p>בדקתי את ההנחה שלך, והיא נכונה. קבצי הקלט זהים (אותו {L('SHA-256')}), הקוד של הרגליים ההגנתיות לא השתנה מאז {L('f9ad358')}, "
        f"והרצה מחדש של {L('a6d.py')} ממטמון ריק נתנה {L('0')} הבדלים ב־{L('1,011')} שדות. לכן חמש האפשרויות נשארות כמו שפורסמו ב־1 באוקטובר.</p>")
    cards = ""
    # The published "more return" row holds a slice of the four-pod monthly book of 2026-10-01; say so on the card.
    def_why = {"fund_defensive_plus": f"{R}{L('60/40')} עם {L('20%')} מהתיק החודשי מ־1 באוקטובר (ארבעה פודים) ו־{L('20%')} {L('cash')}; כפי שפורסם. נקודת קצה, ולכן ציפייה סבירה היא {L('CAGR ~9.2%')}."}
    for k in V4.META["def_cards"]:
        r, m = AR[k], V4.ROWS[k]
        cards += (f'<div class="card d{" pick" if m.get("pick") else ""}"><div><span class="chip {m["chipc"]} ltr">{m["chip"]}</span></div><h3>{rl(m["name"])}</h3>'
                  + pie(r["weights"])
                  + tiles([("CAGR", pct(r["m"]["cagr"])), ("Sharpe", f2(r["m"]["xsharpe"])), ("Max DD", sg(r["m"]["maxdd"]))])
                  + kv([("DD beyond −10%", pct(r["tail"]["p10"])), ("GFC window", sg(r["m"]["crises"]["gfc"])),
                        ("2022 bear window", sg(r["m"]["crises"]["bear_2022"])), ("Needs", m["when"].replace(" wired", " wiring"))])
                  + f'<div class="code ltr">{m["file"]}.yaml</div><ul class="why"><li>{def_why.get(m["file"], m["why"])}</li></ul></div>')
    out["DEF_CARDS"] = cards
    rows = []
    for k in V4.META["def_menu"]:
        if k.startswith("#"):
            rows.append(f'<tr class="grp"><td colspan="15">{k[1:]}</td></tr>')
            continue
        r, m = AR[k], V4.ROWS[k]
        rows.append(tr([name_cell(rl(m["name"]), mix(r["weights"])), td(pct(r["m"]["cagr"])), td(f2(r["m"]["xsharpe"])), td(sg(r["m"]["maxdd"])),
                        td(sg(r["m"]["cvar5_21d"])), td(sg(r["m"]["worst_year"])), td(f2(r["m"]["beta"])), td(pct(r["tail"]["p7"])), td(pct(r["tail"]["p10"])),
                        td(pct(r["frames"]["s3_plus_5bps"]["cagr"])), td(f'{r["ops"]["pods"]:.0f}'), td(f'{r["ops"]["trade_days_per_year"]:.0f}'),
                        td(usd(r["capacity"], r.get("capacity_top"))), lt(", ".join(h["POD"].get(p, p) for p in r["needs_wiring"]) or "none")],
                       "pick" if m.get("pick") else "dim" if m.get("dim") else ""))
    out["DEF_MENU"] = table(["Option", "CAGR", "Excess Sharpe", "Max DD", "CVaR 1M", "Worst year", "Beta", "DD beyond −7%", "DD beyond −10%", "CAGR +5 bps", "Pods",
                             "Trade days/yr", "Capacity", "Needs wiring"], rows)
    out["DEF_CAP"] = notes([
        f"<b>הכלל לא השתנה:</b> {L('Max DD ≥ −7%, DD beyond −10% ≤ 10%')}, גם בעלויות המודל וגם ב־{L('+5 bps')}, ו־2008 ו־2022 לא גרועים מ־{L('−1%')}.",
        f"<b>המספרים:</b> כפי שפורסמו ב־1 באוקטובר (שנה קלנדרית; ההבדל מול שיטת 252 הימים של פרק הצמיחה קטן מ־{L('0.1 pp')}). בכרטיסים {L('Sharpe')} הוא {L('Excess Sharpe')} מעל {L('BIL')}.",
        f"<b>{L('DD beyond −10%')}:</b> חלק המסלולים בבוטסטרפ עם ירידה עמוקה מ־{L('10%')}. זה סרגל, לא הסתברות.",
        f"<b>{L('Capacity')}:</b> נתיב עבודה ובלוקים של מודל הבית, לפני מדידה אמיתית.",
    ])
    if D:
        mr_s, mr_g = D["more_return"]["s9_parity"]["pick"], D["more_return"]["gr1"]["pick"]
        rows = []
        for lab_, p_ in (("יותר תשואה, כפי שפורסם ב־1 באוקטובר (פרוסת הצמיחה הישנה: התיק החודשי של ארבעה פודים)", mr_s), (f"יותר תשואה, עם פרוסת {L('Growth')} (שלב היעד)", mr_g)):
            if p_ is None:
                rows.append(tr([name_cell(lab_, "no passing mix")] + [td("–")] * 9))
                continue
            rows.append(tr([name_cell(lab_, mix(p_["weights"])), td(pct(p_["q"]["cagr"])), td(f2(p_["q"]["xs"])), td(sg(p_["q"]["dd"])), td(sg(p_["q"]["crises"]["gfc"])),
                            td(sg(p_["q"]["crises"]["bear_2022"])), td(pct(p_["tails"]["p10"])), td(pct(p_["tails_plus5"]["p10_max"])), td(pct(p_["frames"]["s3_plus_5bps"]["cagr"])),
                            td(f'{p_["g"] * (1 - p_["cash"]) * 100:.1f}% / {p_["cash"] * 100:.0f}%')]))
        fr_ = D["more_return"]["gr1"]["frontier_launch"]
        t1_ = (table(["Row", "CAGR", "Excess Sharpe", "Max DD", "GFC window", "2022 bear window", "DD beyond −10%", "+5 bps, worst seed", "CAGR +5 bps", "Growth slice / cash, % of capital"], rows)
               + notes([
                   f'<b>נקודת קצה:</b> הבחירה עם {L("Growth")} עוברת את רצפת המשברים ({L("−5%")}) ב־{L(f"{(fr_[0]['worst'] + 0.05) * 100:.2f} pp")} בלבד, והשכנות שלה נכשלות.',
                   f'<b>שלוש הנקודות הראשונות שעוברות:</b> {L(" · ".join(f"growth {x['g'] * (1 - x['cash']) * 100:.0f}% / cash {x['cash'] * 100:.0f}%: CAGR {pct(x['cagr'], 2)}" for x in fr_))}. ההבדל בתשואה ביניהן הוא רעש; מי שרוצה שורה שעדיין נקראת הגנתית צריך את הקטנה.',
                   f'<b>השורה שפורסמה:</b> חושבה עם התיק החודשי מ־1 באוקטובר (ארבעה פודים) כפרוסת הצמיחה, והיא נשארת כך. היא מוצגת כאן בשיטת 252 הימים ({L(pct(mr_s["q"]["cagr"]) if mr_s else "–")}; בכרטיס שנה קלנדרית).',
               ]))
        rows = []
        for tag, lab_ in (("hpi_vote_10 (stored A6-d row)", f"שדרוג מותנה כפי שפורסם ({L('HPI 9%')})"), ("mr_capsule_10", f"שדרוג מותנה עם קפסולת ה־{L('MR')}"),
                          ("hpi_g_10", f"שדרוג מותנה עם {L('HPI-G')} בלבד")):
            g_ = D["gated_upgrade"][tag]
            if not g_.get("pick"):
                rows.append(tr([name_cell(lab_, "no passing cash level")] + [td("–")] * 8))
                continue
            p_, sh = g_["pick"], g_["shares_vs_launch"]
            mx = mix(p_["weights"]).replace("% MR", "% HPI-G") if tag == "hpi_g_10" else mix(p_["weights"])
            rows.append(tr([name_cell(lab_, mx), td(pct(p_["q"]["cagr"])), td(f2(p_["q"]["xs"])), td(sg(p_["q"]["dd"])), td(pct(p_["tails"]["p10"])),
                            td(pct(sh["main"], 1)), td(pct(sh["plus5"], 1)), td(pct(sh["plus10"], 1)), td(yes(g_["vs_launch"]["passed"]))]))
        t2_ = table(["Row", "CAGR", "Excess Sharpe", "Max DD", "DD beyond −10%", "Beats launch: paths", "at +5 bps", "at +10 bps", "All checks"], rows)
        out["DEF_REFRESH"] = (f"<p>שתי השורות נשארות כפי שפורסמו. לידן מוצגת גרסת שלב היעד, שלא מחליפה כלום עד שהקפסולות מחווטות ושער העלות של ה־{L('MR')} נמדד. "
                              f"אותו קוד עם תיק הצמיחה הישן שחזר את השורה שפורסמה בדיוק.</p>" + t1_ + t2_)
    else:
        out["DEF_REFRESH"] = "<p>–</p>"

    # ── GROWTH ──
    s0 = K[S0]
    v0 = V[f"GR1 | {S0}"]["frames"]["main"]
    v0p5 = V[f"GR1 | {S0}"]["frames"]["plus5"]
    tie0 = "תיקו" if 0.21 < v0["share_xs"] < 0.79 else f"{L('40 / 30 / 30')} גבוה יותר" if v0["share_xs"] >= 0.79 else "ההון השווה גבוה יותר"
    em_k = lambda code, key: B["edge_decay"]["edge_margin"][fin[code]]["by_limit"][key]["lowest_passing_k"]  # noqa: E731
    out["GRO_INTRO"] = points([
        f"<b>שלושה מנועים עם מנגנון שונה:</b> {L('TAA')} (הקצאה טקטית, עם {L('TQQQ')} כשהשוק חיובי), מומנטום במניות {L('Nasdaq-100')} מאחורי שערי משטר, והיפוך לממוצע במניות {L('S&amp;P 500')} שנפתח רק בלחץ. "
        f"אף אחד לא יודע איזה מהם יחזיק מעמד, ולכן ברירת המחדל שנרשמה מראש הייתה הון שווה, בלי אומדן של תשואה.",
        f"<b>ההחלטה שלך (5 באוקטובר, אחרי התוצאות):</b> {L('40 / 30 / 30')}, הטיה קלה ל־{L('TAA')}, המנוע הוותיק; {L('TAA 3x')} הוא הפוד שכבר נסחר היום בחשבון שלך. זו נקודה מחוגת ה־{L('TAA')} שהוגדרה מראש, לא חיפוש חדש.",
        f"<b>מול ההון השווה:</b> {L(f'CAGR {pct(g1['q']['cagr'])} / Excess Sharpe {f2(g1['q']['xs'])} / Max DD {sg(g1['q']['dd'])}')} מול {L(f'{pct(s0['q']['cagr'])} / {f2(s0['q']['xs'])} / {sg(s0['q']['dd'])}')}. "
        f"ב־{L('Sharpe')}: {tie0} ({L(pct(v0['share_xs'], 0))} מהמסלולים, {bar(v0['share_xs'])} {L('80%')}). בתשואה {L('40 / 30 / 30')} גבוה יותר ב־{L(pct(v0['share_cagr'], 0))} מהמסלולים. "
        f"ב־{L('+5 bps')} ל־{L('40 / 30 / 30')} יש {L('Sharpe')} גבוה יותר ב־{L(pct(v0p5['share_xs'], 0))} מהמסלולים, כי הוא סוחר פחות {L('MR')}.",
        f"<b>המחיר של ההטיה:</b> יותר תלות ב־{L('TAA')}. אם הוא מת נשאר {L(f'Excess Sharpe {f2(dead('GR1'))}')} במקום {L(f2(dead(S0)))}, ובתרחיש השמרני הירידה היא {L(sg(cons('GR1').get('dd')))} מול {L(sg(cons(S0).get('dd')))}. "
        f"ההבדל ב־{L('DD beyond −20%')} ({L(pct(g1['tails']['p20']))} מול {L(pct(s0['tails']['p20']))}) קטן מרעש הסימולציה.",
        f"<b>לא שלושה הימורים בלתי תלויים:</b> {L('TAA')} והמומנטום שניהם תלויים בנאסד״ק, וביחד הם {L(pct(dep['GR1']['cluster_risk_share']['Nasdaq pair'], 0))} מהסיכון של {L('Growth')}.",
        f"<b>סולם אחד:</b> {L('Growth')} = {L('TAA 3x 40 / 30 / 30')}. {R}{L('Growth Plus')} = אותם משקלים עם {L('TAA 3x 1N')} (רק החלפת הגרסה, שמחזיקה יותר {L('TQQQ')}). {R}{L('Aggressive')} = {L('TAA 3x 1N 60 / 20 / 20')}. הלוויינים תמיד מתחלקים שווה בשווה.",
        (f"<b>כלל התפריט:</b> {not_offered}" if not g3_off else ""),
        f"<b>תיק חודשי אחד:</b> {L('Monthly')} = {L(m_mix)} (החלטה שלך; {L('Monthly Plus')} בוטל). שני פודים, מסחר חודשי בלבד. הוא יכול לרוץ ראשון וגדל עם הקרן."
        + (f" הוא על מדרגת {L(rung_m or 'none')}, כלומר יותר סיכון מ־{L('Growth')}, לא פחות." if more_risk_m else ""),
        f"<b>למה {L('Growth')} נשאר היעד:</b> בתיק של שני פודים יש מנוע תשואה אחד ורגל הגנתית, ואין גיבוי אם ה־{L('TAA')} מפסיק לעבוד.",
        (f"<b>מינוף כחלופה, לא כמוצר:</b> {L('Growth × 1.40')} מגיע לרמת התשואה של {L('Aggressive')} בלי לשנות את האיזון בין המנועים. הוא מוצג בכרטיס נפרד ובפרק המינוף; אפשרי רק עם חשבון מרג׳ין אחד." if LEV else ""),
    ])

    def gcard(code: str) -> str:
        n = fin[code]
        b, p = K[n], S["products"][code]
        hard = HARD[p["target_rung"]]
        capsule_limited = C.get("books", {}).get(n, {}).get("capacity_limited")
        chip = {"GR1": ("go", "FLAGSHIP"), "GR2": ("alt", "SAME WEIGHTS · 1N"),
                "GR3": ("tgt", "MOST RETURN · TAA-LED") if g3_off else ("bad", f"NOT OFFERED · step {step_txt} &lt; 1.5 pp")}[code]
        em = B["edge_decay"]["edge_margin"][n]["by_limit"]
        hk_ = {"GR1": "p20", "GR2": "p25", "GR3": hk3}[code]
        k_note = (em.get(hk_) or {}).get("lowest_passing_k")
        rung_note = ((f"holds down to {k_note:.2f} of the edge" if k_note is not None else "does not hold at any tested share of the edge")
                     + ("" if code != "GR3" else (f"; AGGRESSIVE rung approved by the owner on {AGG_APPROVED}" if AGG_APPROVED else "; AGGRESSIVE needs your approval") if g3_off
                        else "; AGGRESSIVE rung: not decided, moot while not offered"))
        why = {"GR1": [f"הטיה קלה ל־{L('TAA')}, המנוע הוותיק.",
                       f"אף מנוע לא שולט: {L('TAA')} נושא {L(pct(dep['GR1']['risk_share']['TAA'], 0))} מהסיכון.",
                       "בהון שווה התוצאה כמעט זהה."],
               "GR2": [f"אותם משקלים כמו {L('Growth')}.",
                       f"{R}{L('TAA 3x 1N')} במקום {L('TAA 3x')}: יותר {L('TQQQ')}.",
                       f"{R}{L('TAA')} נושא {L(pct(dep['GR2']['risk_share']['TAA'], 0))} מהסיכון."],
               "GR3": [f"{R}{L('60%')} ב־{L('TAA 3x 1N')}: מוצר של מנוע אחד ({L(pct(dep['GR3']['risk_share']['TAA'], 0))} מהסיכון).",
                       ("המחיר: " + ("הירידה העמוקה בסולם, ו" if g3["q"]["dd"] < min(g1["q"]["dd"], g2["q"]["dd"]) else "")
                        + ("החלש ביותר" if (dead("GR3") or 0) < min(dead("GR1") or 0, dead("GR2") or 0) else "חלש")
                        + f" כשה־{L('TAA')} מפסיק לעבוד ({L(f'Excess Sharpe {f2(dead('GR3'))}')}).")]
                      + ([f"על המדרגה החדשה {L('AGGRESSIVE')}" + (" (אושרה)." if AGG_APPROVED else ", שעוד לא אושרה.")] if needs_agg and g3_off else [])
                      + ([] if g3_off else [f"לא מוצע: רק {L(step_txt)} מעל {L('Growth Plus')}, ונדרש {L('1.5 pp')}."])
                      + [f"החלופה המתונה: {L('Growth Plus')}."]}[code]
        why = '<ul class="why">' + "".join(f"<li>{x}</li>" for x in why) + "</ul>"
        rows_ = []
        if code == "GR3":
            rows_.append(("DD beyond −25%", pct(b["tails"]["p25"])))
        rows_ += [(f"DD beyond −{hard[1:]}%", pct(b["tails"][hard])), ("GFC window", sg(b["q"]["crises"]["gfc"])), ("2022 bear window", sg(b["q"]["crises"]["bear_2022"])),
                  ("Rung passed", p["strictest_rung"] or "none"), (rung_note, None),
                  ("Capacity, all at the open", capstr(n, "MOO")), ("Capacity, MR at the close", "up to " + capstr(n)),
                  ("upper bound, not modelled; limit: MR stocks (FOX, NWS)" if capsule_limited else "", None),
                  ("Needs", "capsule wiring + MR cost gate")]
        fname = {"GR1": "fund_growth", "GR2": "fund_growth_plus", "GR3": "fund_growth_aggressive"}[code]
        return (f'<div class="card g{" pick" if code == "GR1" else ""}"><div><span class="chip {chip[0]} ltr">{chip[1]}</span></div><h3>{GR[code][1]} <span class="ltr cap">{GR[code][0]}</span></h3>'
                + pie(b["weights"])
                + tiles([("CAGR", pct(b["q"]["cagr"])), ("Sharpe", f2(b["q"]["xs"])), ("Max DD", sg(b["q"]["dd"]))])
                + kv(rows_) + f'<div class="code ltr">{fname}.yaml</div>{why}</div>')

    def mcard(key: str, he: str, en_: str, chip: str, fname: str, why: list) -> str:
        """A monthly book (two pods) in the same card layout as the capsule products."""
        b = K[key]
        hard = HARD.get(b.get("strictest_rung") or "", "p20")
        ez = C.get("books", {}).get(key, {}).get("ease", {})
        need = ", ".join(h["POD"].get(a, a) for a in ez.get("not_wired", []))
        return (f'<div class="card m"><div><span class="chip stage ltr">{chip}</span></div><h3>{he} <span class="ltr cap">{en_}</span></h3>'
                + pie(b["weights"])
                + tiles([("CAGR", pct(b["q"]["cagr"])), ("Sharpe", f2(b["q"]["xs"])), ("Max DD", sg(b["q"]["dd"]))])
                + kv([(f"DD beyond −{hard[1:]}%", pct(b["tails"][hard])), ("GFC window", sg(b["q"]["crises"]["gfc"])),
                      ("2022 bear window", sg(b["q"]["crises"]["bear_2022"])), ("Rung passed", b.get("strictest_rung") or "none"),
                      ((f"holds down to {margin_k(key):.2f} of the edge; " if margin_k(key) is not None else "")
                       + f"at 3/4 of the edge: {pct(cons_breach(key))} beyond −{hard[1:]}% against the 15% cap", None),
                      ("Capacity, all at the open", capstr(key, "MOO")), ("Capacity, worked", capstr(key)), ("monthly orders, worked over days", None),
                      ("Needs", (need + " wiring") if need else "nothing" if ez else "–")])
                + f'<div class="code ltr">{fname}.yaml</div><ul class="why">' + "".join(f"<li>{x}</li>" for x in why) + "</ul></div>")

    def lcard() -> str:
        """Growth at a fixed 1.40x: an alternative route to Aggressive's level of return, not a product on the menu."""
        if not LEV:
            return ""
        q, t, nm = LEV["q"], LEV.get("tails", {}), LEV.get("name", "")
        cz = q.get("crises") or (K.get(nm, {}).get("q", {}).get("crises") or {})
        xl_ = (X["books"].get(nm) or {}).get("nasdaq_lookthrough", {}).get("max")
        rows_ = [("DD beyond −25%", pct(t.get("p25"))), ("DD beyond −30%", pct(t.get("p30"))), ("GFC window", sg(cz.get("gfc"))), ("2022 bear window", sg(cz.get("bear_2022"))),
                 ("CAGR at +5 bps", pct((LEV.get("plus5") or {}).get("cagr"))), ("CAGR at a 2.5% financing spread", pct((LEV.get("spread_250") or {}).get("cagr"))),
                 ("Nasdaq look-through, peak", f"{xl_:.2f}×" if xl_ else "–"), ("Excess Sharpe, TAA dead", f2(dead(nm))),
                 ("Capacity", "Growth's, divided by 1.40"), ("Needs", "one cross-margined account (fund stage)")]
        why = [f"שומר על האיזון בין שלושת המנועים, במקום לרכז את הסיכון ב־{L('TAA')}.",
               f"העלות: מימון ב־{L('DTB3 + 1.5%')}; בלי קריאות מרג׳ין ובלי יום פער במודל; המינוף ממנף גם את עלות המסחר של ה־{L('MR')}, והגודל המרבי קטן פי {L('1.4')}.",
               f"לא זמין היום: כל פוד יושב בחשבון {L('Reg-T')} נפרד."]
        return ('<div class="card m"><div><span class="chip stage ltr">ALTERNATIVE · LEVERAGE 1.40×</span></div><h3>צמיחה במינוף <span class="ltr cap">Growth × 1.40</span></h3>'
                + pie(g1["weights"]) + '<div class="kv"><div class="kvn">the weights of Growth, each × 1.40; financed by a margin loan</div></div>'
                + tiles([("CAGR", pct(q["cagr"])), ("Sharpe", f2(q["xs"])), ("Max DD", sg(q["dd"]))])
                + kv(rows_) + '<div class="code ltr">not a product: no YAML</div><ul class="why">' + "".join(f"<li>{x}</li>" for x in why) + "</ul></div>")

    out["GRO_CARDS"] = ("".join(gcard(c) for c in GR) + lcard()
                        + mcard(S9, "החודשי", "Monthly", "RUNS FIRST · SCALES", "fund_growth_monthly",
                                ["שני פודים, מסחר חודשי בלבד.", f"יכול לרוץ ראשון: חסר רק חיווט של {L('CORE5')}.",
                                 f"המחיר: מנוע תשואה אחד, בלי גיבוי. עם {L('TAA')} מת נשאר {L(f'Excess Sharpe {f2(d_s9)}')}."]
                                + ([f"על מדרגת {L(rung_m or 'none')}: יותר סיכון מ־{L('Growth')}, לא פחות."] if more_risk_m else [])
                                + (["המרווח במדרגה דק."] if thin_m else [])))

    def grow(n: str, he: str = "", cls: str = "") -> str:
        b = K[n]
        t = b["tails"]
        return tr([name_cell(he or en(n), mix(b["weights"]) if b["weights"] else ""), td(pct(b["q"]["cagr"])), td(f2(b["q"]["xs"])), td(pct(b["q"]["vol"])), td(sg(b["q"]["dd"])),
                   td(pct(t["p20"])), td(pct(t["p25"])), td(sg(b["q"]["crises"]["gfc"])), td(sg(b["q"]["crises"]["bear_2022"])), td(sg(b["q"]["worst_year"])),
                   td(pct(b["frames"]["s3_plus_5bps"]["cagr"])), td(pct(b["frames"]["s1_house_cash"]["cagr"])), lt(b["strictest_rung"] or "none"), td(capstr(n))], cls)

    bench_row = lambda k, lab_: tr([name_cell(lab_)] + [td(pct(S["bench"][k]["cagr"])), td(f2(S["bench"][k]["xs"])), td(pct(S["bench"][k]["vol"])),  # noqa: E731
                                                       td(sg(S["bench"][k]["dd"]))] + [td("–")] * 9, "dim")
    grp = lambda s: f'<tr class="grp"><td colspan="15">{s}</td></tr>'  # noqa: E731
    rows = [grp("Capsule products"), grow(fin["GR1"], GR["GR1"][1], "pick"), grow(fin["GR2"], GR["GR2"][1]), grow(fin["GR3"], GR["GR3"][1]),
            grp("Monthly books"), grow(S9, "החודשי")]
    rows += [grow(k_, he_, "dim") for k_, he_ in ((S13, "החודשי מ־1 באוקטובר, ארבעה פודים (להשוואה)"), (OLDP, "החודשי עם יותר תשואה, מ־1 באוקטובר, ארבעה פודים (להשוואה)")) if k_ in K]
    rows += [grow(GR1_L, f"{R}{L('Growth')} בלי {L('MR')} (ה־{L('30%')} שלו ב־{L('BIL')})"),
             grp("Registered default"), grow(S0, "הון שווה, שליש לכל קפסולה (ברירת המחדל הרשומה)"),
             grp("Each capsule alone"), grow("capsule TAA 3x"), grow("capsule TAA 3x 1N"), grow("capsule MOM"), grow("capsule MR"),
             grp("Benchmarks"), bench_row("SPXTR", "S&amp;P 500 TR"), bench_row("QQQ", "QQQ TR")]
    out["GRO_MENU"] = table(["Option", "CAGR", "Excess Sharpe", "Vol", "Max DD", "DD beyond −20%", "DD beyond −25%", "GFC window", "2022 bear window",
                             "Worst calendar year", "CAGR +5 bps", "CAGR cash 0%", "Rung passed", "Capacity (worked route, upper bound)"], rows)
    sm = B["reset"]["GR1"]["start_month_spread"]
    c3 = cons(fin["GR3"]).get(hk3)
    blk3 = B["bootstrap"]["blocks"]
    b3 = lambda k_: (blk3.get(k_, {}).get(fin["GR3"]) or {}).get(hk3)  # noqa: E731
    out["GRO_CAP"] = notes([
        f"<b>קיצורים בטבלאות:</b> {L('MOM = momentum capsule, MR = mean-reversion capsule')}.",
        f"<b>התיק החודשי והמדרגה:</b> הוא עובר את {L(rung_m or 'none')}" + (f" ולא את {L('GROWTH')} ({L(pct(s9['tails'].get('p20')))} מהמסלולים מעבר ל־{L('−20%')})" if more_risk_m else "")
        + f". {monthly_margin}. ב־{L('Growth')} המדרגה מחזיקה עד {L(f2(em_k('GR1', 'p20')))} מהיתרון.",
        f"<b>כל המספרים הם ה־{L('backtest')}:</b> עמלות {L('IBKR')} ו־{L('2.5 bps')} החלקה לצד כבר בפנים, ומזומן פנוי מרוויח ריבית. התרחיש השמרני ותרחיש הלחץ נמצאים בפרק ״כמה להאמין״. בכרטיסים {L('Sharpe')} הוא {L('Excess Sharpe')} מעל {L('BIL')}.",
        f"<b>{L('DD beyond −X%')}:</b> חלק המסלולים בבוטסטרפ עם ירידה עמוקה מ־X. זה סרגל (בלוק 63, יתרון מלא), לא הסתברות.",
        f"<b>{L('GFC window')} ו־{L('2022 bear window')}:</b> תשואה מנקודה לנקודה על חלון המשבר ({L('5/2008 → 3/2009')}, {L('1/2022 → 10/2022')}), לא שנה קלנדרית.",
        f"<b>{L('Aggressive')} על המדרגה שלו ({L(rung3 or 'none')}, גבול {L('−' + hk3[1:] + '%')}):</b> מספר החריגה {L(pct(g3['tails'].get(hk3)))}; המדרגה מחזיקה עד {L(fk(em3))} מהיתרון. "
        + (f"בתרחיש השמרני: {L(pct(c3))} מול תקרה של {L('15%')}. " if c3 is not None else "")
        + (f"בבלוק 21: {L(pct(b3('21')))}; בימים בלתי תלויים: {L(pct(b3('1')))}. " if b3("21") is not None else "")
        + (f"ב־{L('−25%')} (הגבול של {L('GROWTH PLUS')}) הוא {L(pct(g3['tails'].get('p25')))}, ו־{L('Max DD')} שלו {L(sg(g3['q']['dd']))}." if hk3 != "p25" else f"{R}{L('Growth Plus')} מחזיק באותה מדרגה עד {L(fk(em2))}."),
        (f"<b>{L('Aggressive')}:</b> {not_offered_ref}" if not g3_off else ""),
        (f"<b>המדרגה {L('AGGRESSIVE')}:</b> {agg_ok_txt} {R}{L('Aggressive')} עובר רק אותה." if needs_agg and g3_off and AGG_APPROVED
         else f"<b>המדרגה {L('AGGRESSIVE (−27% / −30%)')}:</b> חדשה" + (", ומחכה לאישור שלך." if g3_off else f". כל עוד {L('Aggressive')} לא מוצע, אין צורך להכריע עליה.")),
        f"<b>תאריך האיפוס:</b> האיפוס השנתי הוא בינואר. החציון על 12 חודשי התחלה ב־{L('Growth')} הוא {L(f'{pct(sm['cagr']['median_of_12'], 2)} / {f2(sm['xs']['median_of_12'])}')}, מול {L(f'{pct(sm['cagr']['january'], 2)} / {f2(sm['xs']['january'])}')} בינואר (מספר הכותרת).",
        f"<b>שורות ההשוואה מ־1 באוקטובר:</b> התיקים החודשיים הקודמים, של ארבעה פודים ({L('TAA 3x 1N, NDX-VXN, CORE5, BTAL_QQQ')}). הם לא מוצעים יותר. לתיק הישן עם יותר תשואה לא חושבה קיבולת.",
    ])

    # ── why two pods in the monthly book (monthly.json; exploratory, run after the results) ──
    if M.get("ladder"):
        ok = lambda r: yes(r if isinstance(r, bool) else (r or {}).get("pass"))  # noqa: E731
        okb = lambda r: bool(r if isinstance(r, bool) else (r or {}).get("pass"))  # noqa: E731
        mrow = lambda lab_, r, cls="": tr([lt(lab_), td(pct(r["q"]["cagr"])), td(f2(r["q"]["xs"])), td(pct(r["q"]["vol"])), td(sg(r["q"]["dd"])), td(pct(r["tails"].get("p20"))),  # noqa: E731
                                           td(pct(r["tails"].get("p25"))), td(sg(r["q"]["crises"]["gfc"])), td(sg(r["q"]["crises"]["bear_2022"])), td(pct(r["plus5"]["cagr"])),
                                           td(ok(r["rung_growth"])), td(ok(r["rung_growth_plus"]))], cls)
        g7 = lambda s: f'<tr class="grp"><td colspan="12">{s}</td></tr>'  # noqa: E731
        prod_lab = {f"1N {mw.get('taa3x_1n', 0) * 100:.0f}% / CORE5 {mw.get('core5', 0) * 100:.0f}%": " = Monthly"}
        rows_m = [g7("TAA 3x 1N / CORE5 ladder (no momentum)")]
        lev_hdr = False
        for nm, r in M["ladder"].items():
            if "L" in r and not lev_hdr:
                rows_m.append(g7("Leverage on a calmer mix (margin loan)"))
                lev_hdr = True
            rows_m.append(mrow(f"TAA 3x {nm}" + prod_lab.get(nm, ""), r, "pick" if nm in prod_lab else ""))
        rows_m.append(g7("Reference books"))
        for nm, r in M["reference"].items():
            rows_m.append(mrow(show(nm), r, "dim" if "2026-10-01" in nm else ""))
        t_lad = table(["Monthly book", "CAGR", "Excess Sharpe", "Vol", "Max DD", "DD beyond −20%", "DD beyond −25%", "GFC window", "2022 bear window", "CAGR +5 bps", "GROWTH rung", "GROWTH PLUS rung"], rows_m)
        G = M["grid"]
        gkey = lambda taa, t, m_: f"{taa} {t:.0%}:{1 - t:.0%} mom {m_:.0%}"  # noqa: E731
        rows_g = []
        for taa in ("taa3x_1n", "taa3x"):
            for t_ in (0.5, 0.7):
                b0, b30 = G.get(gkey(taa, t_, 0.0)), G.get(gkey(taa, t_, 0.30))
                if not b0 or not b30:
                    continue
                tn = "TAA 3x 1N" if taa == "taa3x_1n" else "TAA 3x"
                for lab_, r in ((f"{tn} {t_ * 100:.0f} / CORE5 {(1 - t_) * 100:.0f}, no momentum", b0), (f"the same with momentum 30%", b30)):
                    v_ = r.get("vs_no_momentum")
                    rows_g.append(tr([lt(lab_), td(pct(r["q"]["cagr"])), td(f2(r["q"]["xs"])), td(sg(r["q"]["dd"])), td(sg(r["q"]["crises"]["bear_2022"])),
                                      td(pct(v_["share_xs"], 0) if v_ else "–"), td(pct(v_["share_cagr"], 0) if v_ else "–")]))
        t_mom = table(["Does momentum earn a weight", "CAGR", "Excess Sharpe", "Max DD", "2022 bear window", "Higher Excess Sharpe than without momentum: paths", "Higher CAGR: paths"], rows_g)
        cr = M.get("corr", {})
        cr_lab = {"btal_qqq | taa3x_1n": "BTAL_QQQ and TAA 3x 1N", "core5 | taa3x_1n": "CORE5 and TAA 3x 1N", "core5 | taa3x": "CORE5 and TAA 3x", "core5 | MOM": "CORE5 and the momentum capsule",
                  "ndx_vxn | MOM": "NDX-VXN and the momentum capsule"}
        t_cr = table(["Daily correlation"] + [cr_lab.get(k_, k_) for k_ in cr], [tr([lt("2008-03 → 2026-08")] + [td(f2(v_)) for v_ in cr.values()])])
        withm = [r for r in G.values() if r.get("vs_no_momentum")]
        sh = [r["vs_no_momentum"]["share_xs"] for r in withm]
        with1n = [r for r in withm if r["taa"] == "taa3x_1n"]
        n1_below = sum(r["vs_no_momentum"]["share_xs"] < 0.5 for r in with1n)
        base_of = lambda r: G[gkey(r["taa"], r["taa_to_core5"], 0.0)]  # noqa: E731
        n_cagr_le = sum(r["q"]["cagr"] <= base_of(r)["q"]["cagr"] + 0.0005 for r in withm)
        dd_gain = [(r["q"]["dd"] - base_of(r)["q"]["dd"]) * 100 for r in withm]
        n_22_worse = sum(r["q"]["crises"]["bear_2022"] < base_of(r)["q"]["crises"]["bear_2022"] for r in withm)
        noc = M["reference"].get("TAA 3x 57 / momentum 43, no CORE5")
        oldm = M["reference"].get("old monthly (2026-10-01)")
        lad = M["ladder"]
        plain = {k_: v_ for k_, v_ in lad.items() if "L" not in v_}
        first20 = next((k_ for k_, v_ in plain.items() if v_["q"]["cagr"] >= 0.20), None)
        levs = {k_: v_ for k_, v_ in lad.items() if "L" in v_}
        vm = V.get(f"{S9} | {S13}", {}).get("frames", {}).get("main")
        # leverage on a calmer mix against the unlevered ladder point of about the same return: better only if the drawdown
        # is shallower by more than 1 pp at a CAGR not lower by more than 0.3 pp
        lev_read = ""
        if levs and first20:
            ref_ = plain[first20]["q"]
            better_ = [v_["q"]["dd"] - ref_["dd"] > 0.01 and v_["q"]["cagr"] - ref_["cagr"] > -0.003 for v_ in levs.values()]
            worse_ = [v_["q"]["dd"] - ref_["dd"] < -0.01 or v_["q"]["cagr"] - ref_["cagr"] < -0.01 for v_ in levs.values()]
            lev_read = ("המינוף נותן ירידה רדודה יותר באותה תשואה; שווה בדיקה נפרדת." if any(better_) and not any(worse_)
                        else "המינוף נותן תוצאה דומה (הבדלים בתוך הרעש), ודורש חשבון מרג׳ין; לכן אין סיבה למנף." if not any(better_) and not any(worse_)
                        else "המינוף לא נותן תוצאה טובה יותר, ודורש חשבון מרג׳ין; לכן אין סיבה למנף.")
        dd_txt = (f"ירידה היסטורית רדודה יותר ב־{L(f'{min(dd_gain):.1f} … {max(dd_gain):.1f} pp')}" if dd_gain and min(dd_gain) > 0
                  else f"שינוי ב־{L('Max DD')} ההיסטורי של {L(f'{min(dd_gain):+.1f} … {max(dd_gain):+.1f} pp'.replace('-', '−'))}")
        solo = K["capsule TAA 3x 1N"]
        why2 = [
            f"<b>למה בלי {L('BTAL_QQQ')}:</b> הוא כמעט אותו מנוע כמו {L('TAA 3x 1N')} (מתאם יומי {L(f2(cr.get('btal_qqq | taa3x_1n')))}), ולכן הוא לא מוסיף פיזור. "
            f"המתאם של {L('CORE5')} עם {L('TAA 3x 1N')} הוא {L(f2(cr.get('core5 | taa3x_1n')))}.",
            f"<b>למה בלי מומנטום:</b> קפסולת המומנטום מחליפה את {L('NDX-VXN')} במוצרי הצמיחה, ובתיק חודשי של {L('TAA')} ו־{L('CORE5')} היא לא מרוויחה משקל. ב־{L(str(len(withm)))} תיקי הרשת עם מומנטום, "
            f"ה־{L('Sharpe')} גבוה יותר מאשר בלעדיו ב־{L(f'{pct(min(sh), 0)} … {pct(max(sh), 0)}')} מהמסלולים" + (f", אף פעם לא {L('80%')}" if max(sh) < 0.79 else "")
            + f"; עם {L('1N')}, ב־{L(str(n1_below))} מתוך {L(str(len(with1n)))} זה מתחת ל־{L('50%')}.",
            f"<b>מה המומנטום כן משנה:</b> ה־{L('CAGR')} שווה או נמוך יותר ב־{L(str(n_cagr_le))} מתוך {L(str(len(withm)))}; {dd_txt}; וחלון 2022 גרוע יותר ב־{L(str(n_22_worse))} מתוך {L(str(len(withm)))}.",
            f"<b>למה צריך את {L('CORE5')}:</b> {L('TAA 3x 1N')} לבדו: {L(f'Max DD {sg(solo['q']['dd'])}, DD beyond −20% {pct(solo['tails']['p20'])}')}, ולא עובר אף מדרגה."
            + (f" גם {L('TAA 3x 57 / momentum 43')} בלי {L('CORE5')} " + ("נכשל" if not okb(noc["rung_growth"]) else "עובר") + f" ב־{L('GROWTH')}: מספר החריגה שלו {L(pct(noc['tails'].get('p20')))} מול תקרה של {L('15%')} ({L(f'Max DD {sg(noc['q']['dd'])}')})." if noc else ""),
            (f"<b>מול התיק הישן של ארבעה פודים:</b> {L(f'CAGR {pct(s9['q']['cagr'])} / Excess Sharpe {f2(s9['q']['xs'])} / Max DD {sg(s9['q']['dd'])}')} בחדש, מול {L(f'{pct(oldm['q']['cagr'])} / {f2(oldm['q']['xs'])} / {sg(oldm['q']['dd'])}')} בישן. "
             + (f"ב־{L('Sharpe')} החדש גבוה ב־{L(pct(vm['share_xs'], 0))} מהמסלולים (מתחת לרף, כלומר תיקו); ב־{L('CAGR')} הישן גבוה ב־{L(pct(1 - vm['share_cagr'], 0))}. " if vm else "")
             + "כמעט אותו תיק, עם שני פודים במקום ארבעה." if oldm else ""),
            f"<b>למה {L(m_short)}:</b> החלטה שלך. שורות הסולם בטבלה ({L(' · '.join(k_.replace('1N ', '').replace(' / CORE5 ', ' / ').replace('%', '') for k_ in plain))}) מראות את המחיר: כל צעד מוסיף תשואה ומעמיק את הירידה. "
            + f"ב־{L(m_short)} התיק עובר את {L(rung_m or 'none')}" + (f" ולא את {L('GROWTH')}." if more_risk_m else "."),
            (f"<b>מינוף על תערובת רגועה יותר:</b> {L(' · '.join(f'{k_}: CAGR {pct(v_['q']['cagr'])}, Max DD {sg(v_['q']['dd'])}' for k_, v_ in levs.items()))}. התוצאה דומה לזו של יותר {L('TAA 3x 1N')} בלי מינוף, והיא דורשת חשבון מרג׳ין; לכן אין סיבה למנף כאן." if levs else ""),
            f"<b>מרווח המדרגה:</b> {monthly_margin}." + (" זה מרווח דק יותר מבכל מוצר קפסולות." if thin_m else ""),
            "<b>זו רשת גישוש, לא מבחן:</b> היא הורצה אחרי התוצאות, לבקשתך, ולא נרשמה מראש. ההבדלים בין שורות שכנות קטנים.",
        ]
        out["MONTHLY_WHY"] = notes(why2) + t_lad + t_mom + t_cr
    else:
        out["MONTHLY_WHY"] = f"<p>{R}{L('pending')}: הרשת של התיקים החודשיים ({L('monthly.json')}) עוד לא חושבה.</p>"

    def frow(n: str, he: str = "") -> str:
        b = K[n]
        lg, fr = b["long"], b["frames"]
        return tr([name_cell(he or en(n)), td(f2(b["q"]["sharpe0"])), td(f2(b["q"]["xs_monthly"])), td(f2(b["q"]["sortino"])), td(f2(lg["calmar"])), td(sg(lg["cvar5_21d"])),
                   td(sg(lg["worst_month"])), td(sg(lg["worst_12m"])), td(f2(lg["beta"])), td(f2(lg["crisis_corr"])), td(pct(lg["pos_months"], 0)),
                   td(f'{lg["underwater_days"]:.0f}'), td(pct(b["tails"]["p30"])), td(pct(fr["plus_10bps"]["cagr"])), td(f'{pct(fr["s6_exact"]["cagr"])} / {f2(fr["s6_exact"]["xs"])}'),
                   td(f'{f2(b["halves_xs"][0])} / {f2(b["halves_xs"][1])}'), td(pct(fr["s3b_plus_5bps_ex_bil"]["cagr"])), td(pct(fr["s9_mr_fair_cash"]["cagr"])),
                   td(pct(fr["s10_mr_cash_0"]["cagr"])), td(pct(fr["s2_proxy_unscaled"]["cagr"])), td(sg(b["net"]["dd"]))])
    out["GRO_FULL"] = table(["Option", "Sharpe (rf 0)", "Excess Sharpe, monthly", "Sortino", "Calmar", "CVaR 1M", "Worst month", "Worst 12M", "Beta", "Down-day corr",
                             "Positive months", "Longest underwater (days)", "DD beyond −30%", "CAGR +10 bps", "2012+ CAGR / xs", "Excess Sharpe H1 / H2",
                             "CAGR +5 bps, stocks only", "CAGR, MR fair cash", "CAGR, MR cash 0%", "CAGR, unscaled proxy", "Net Max DD"],
                            [frow(fin["GR1"], GR["GR1"][1]), frow(fin["GR2"], GR["GR2"][1]), frow(fin["GR3"], GR["GR3"][1]), frow(S9, "החודשי"), frow("capsule TAA 3x"), frow("capsule TAA 3x 1N"), frow("capsule MOM"), frow("capsule MR")])

    blk_lab = {"A": "A: 2008-03 → 2012-10 (proxy)", "B": "B: 2012-10 → 2021", "C": "C: 2022 → 2026-08", "RECENT": "Last 3 years"}
    rows_p = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")):
        if "blocks" in K[n]:
            rows_p.append(tr([name_cell(he)] + [td(f'{pct(K[n]["blocks"][k]["cagr"])} / {f2(K[n]["blocks"][k]["xs"])} / {sg(K[n]["blocks"][k]["dd"])}') for k in blk_lab]))
    rows_v = []
    for a_, b_, lab_ in (("GR1", S9, "Growth vs Monthly"), ("GR2", S9, "Growth Plus vs Monthly"), ("GR3", S9, "Aggressive vs Monthly"),
                         ("GR2", "GR1", "Growth Plus vs Growth"), (S9, S13, "Monthly vs the four-pod Monthly of 2026-10-01")):
        v = V.get(f"{a_} | {b_}")
        if not v:
            continue
        rows_v.append(tr([lt(lab_)] + [td(f'{pct(v["frames"][f]["share_xs"], 0)} / {pct(v["frames"][f]["share_cagr"], 0)}') for f in ("main", "plus5", "plus10")]
                         + [td(f'{pct(v["blocks"][k]["main"]["share_xs"], 0)} / {pct(v["blocks"][k]["main"]["share_cagr"], 0)}') for k in blk_lab]))
    bC = {n: K[n]["blocks"]["C"] for n in ("GR1", S9)}
    bB = {n: K[n]["blocks"]["B"] for n in ("GR1", S9)}
    rev_ok = bool(rev.get("passed"))
    out["VERSUS"] = (points([
        f"<b>ההשוואה:</b> {L('Growth')} (שלוש קפסולות, מסחר יומי) מול {L('Monthly')} ({L(m_mix)}, שני פודים). היא מראה מה קונים במעבר מהתיק שרץ ראשון לתיק היעד.",
        f"<b>לפי עלות:</b> {L('Growth')} גבוה ב־{L('Sharpe')} ב־{by_cost}. בתשואה הוא גבוה ב־{L(pct(vf['main']['share_cagr'], 0))} מהמסלולים בעלויות המודל. "
        + ("בעלויות המודל הוא עובר את כל שבע הבדיקות שנקבעו מראש מול החודשי." if rev_ok else "הוא לא עובר את כל שבע הבדיקות שנקבעו מראש מול החודשי."),
        f"<b>לפי תקופה:</b> {by_block}. ב־2012 עד 2021: {L(f'Excess Sharpe {f2(bB['GR1']['xs'])}')} ל־{L('Growth')} מול {L(f2(bB[S9]['xs']))} לחודשי. "
        f"מ־2022: {L(f'{f2(bC['GR1']['xs'])}')} מול {L(f2(bC[S9]['xs']))}, ו־{L(f'CAGR {pct(bC['GR1']['cagr'])}')} מול {L(pct(bC[S9]['cagr']))}. כלומר מ־2022: {since22}.",
        f"<b>אם ה־{L('TAA')} מפסיק לעבוד:</b> {L(f'Excess Sharpe {f2(d_g1)}')} ב־{L('Growth')} מול {L(f2(d_s9))} בחודשי. זה מה שהמנועים הנוספים קונים.",
    ])
                     + table(["Share of paired paths: higher Excess Sharpe / higher CAGR", "Model costs", "+5 bps", "+10 bps"] + list(blk_lab.values()), rows_v)
                     + table(["By sub-period: CAGR / Excess Sharpe / Max DD"] + list(blk_lab.values()), rows_p)
                     + f'<p class="cap">{R}{L("+10 bps")} הוא אקסטרפולציה ליניארית מ־{L("+5 bps")} ברמת התיק. 20,000 מסלולים זוגיים, בלוק 63.</p>')

    # ── what is inside ──
    XB, tq = X["books"], X["tqqq_weight"]
    out["STRUCT_INTRO"] = points([
        f"<b>{L('TQQQ')} בתוך ה־{L('TAA')}:</b> {L('TAA 3x')} מחזיק {L('TQQQ')} בממוצע {L(pct(tq['taa3x']['daily']['mean'], 0))} מהפוד ועד {L('100%')}; גרסת {L('1N')} בממוצע {L(pct(tq['taa3x_1n']['daily']['mean'], 0))}.",
        f"<b>חשיפה לנאסד״ק:</b> דרך {L('TQQQ')} והמומנטום ביחד היא מגיעה בשיא ל־{L(f'{XB['GR1']['nasdaq_lookthrough']['max']:.2f}×')} מהתיק ב־{L('Growth')}, ל־{L(f'{XB['GR2']['nasdaq_lookthrough']['max']:.2f}×')} ב־{L('Growth Plus')} ול־{L(f'{XB['GR3']['nasdaq_lookthrough']['max']:.2f}×')} ב־{L('Aggressive')}, במשקלי היעד "
        f"(עם הסחיפה בין איפוסים: {L(f'{XB['GR1']['nasdaq_lookthrough_drift']['max']:.2f}×')}, {L(f'{XB['GR2']['nasdaq_lookthrough_drift']['max']:.2f}×')} ו־{L(f'{XB['GR3']['nasdaq_lookthrough_drift']['max']:.2f}×')}; נמדד מ־{L('10/2012')}).",
        f"<b>התיק לא מושקע עד הסוף:</b> בממוצע {L(pct(XB['GR1']['tbill_like_share']['mean'], 0))} מ־{L('Growth')} יושב ב־{L('BIL')} או במזומן.",
    ])
    rows = []
    for code in GR:
        n, d_, x_ = fin[code], dep[code], XB[code]
        w = K[n]["weights"]
        capw = {"TAA": w.get("taa3x", 0) + w.get("taa3x_1n", 0), "MOM": w.get("ndx_atr_cap", 0) * 2, "MR": w.get("dv2_g", 0) * 2}
        tk = "taa3x" if "taa3x" in w else "taa3x_1n"
        rows.append(tr([name_cell(GR[code][1]), td(" / ".join(f"{capw[c] * 100:.0f}" for c in ("TAA", "MOM", "MR"))),
                        td(" / ".join(f"{d_['risk_share'][c] * 100:.0f}" for c in ("TAA", "MOM", "MR"))), td(pct(d_["cluster_risk_share"]["Nasdaq pair"], 0)),
                        td(" / ".join(f"{d_['excess_return_share'][c] * 100:.0f}" for c in ("TAA", "MOM", "MR"))), td(f2(d_["enb"]["full"])),
                        td(f'{pct(tq[tk]["daily"]["mean"], 0)} / {pct(tq[tk]["daily"]["p90"], 0)} / {pct(tq[tk]["daily"]["max"], 0)}'),
                        td(f'{x_["nasdaq_lookthrough"]["mean"]:.2f}× / {x_["nasdaq_lookthrough"]["p90"]:.2f}× / {x_["nasdaq_lookthrough"]["max"]:.2f}×'),
                        td(pct(x_["tbill_like_share"]["mean"], 0))]))
    out["STRUCT"] = table(["Product", "Capital % TAA / MOM / MR", "Risk % TAA / MOM / MR", "Nasdaq pair, share of risk", "Excess return % TAA / MOM / MR", "Effective bets",
                           "TQQQ in the TAA pod: mean / P90 / max", "Nasdaq look-through: mean / P90 / peak", "T-bill-like share, mean"], rows)
    dr = {c: XB[c]["nasdaq_lookthrough_drift"] for c in GR}
    out["STRUCT"] += notes([
        f'<b>מאיפה המספרים:</b> החשיפה מחושבת מ־{L("10/2012")} (אין סדרת מחיר ל־{L("TQQQ")} הסינתטי לפני כן), במשקלי היעד.',
        f'<b>עם הסחיפה בין איפוס לאיפוס</b> השיא היה גבוה יותר: {L(" / ".join(f"{dr[c]['max']:.2f}×" for c in GR))} ({L(dr["GR1"]["peak_date"][:4])}). מאז 2015 המקסימום הוא {L(" / ".join(f"{dr[c]['max_since_2015']:.2f}×" for c in GR))}.',
        f'<b>מה לא נבחן:</b> הימים הגרועים ביותר במדגם ({L("−4%")} עד {L("−5%")} ביום) קרו בחשיפה נמוכה, ולכן תאי ה־{L("P90")} והשיא בטבלה למטה לא נבחנו בפועל.',
    ])
    rows = []
    for code in GR:
        gt = XB[code]["gap_table"]
        rows.append(tr([name_cell(GR[code][1])] + [td(f'{sg(gt[f]["mean"])} / {sg(gt[f]["p90"])} / {sg(gt[f]["peak"])}', "neg") for f in ("5%", "10%", "15%", "20%")]))
    out["GAP"] = (points([
        f"<b>יום פער הוא הסיכון שהבדיקה ההיסטורית לא מכילה:</b> אין בה יום של {L('−20%')} בנאסד״ק, וגם לא דוב כמו 2000 עד 2002.",
        f"<b>הטבלה היא חשבון, לא סימולציה:</b> {L('TQQQ')} יורד פי שלושה מהמדד; המומנטום והמניות של ה־{L('MR')} יורדים כמו המדד.",
    ])
                  + table(["Product loss on a one-day Nasdaq-100 fall of", "5%: at mean / P90 / peak joint exposure", "10%", "15%", "20%"], rows))

    yrs_x = sorted(XB["GR1"]["nasdaq_lookthrough"]["by_year_max"])
    out["GAP"] += ("<details><summary>חשיפה לנאסד״ק לפי שנה (שיא)</summary>"
                   + table(["Product"] + yrs_x, [tr([name_cell(GR[c][1])] + [td(f'{XB[c]["nasdaq_lookthrough"]["by_year_max"][y]:.2f}×') for y in yrs_x]) for c in GR]) + "</details>")
    # ── the blend dial between Defensive and Growth (study.json blends; descriptive) ──
    BL = S["blends"]
    cg = S["corr_gr1_defensive"]
    crz = lambda q, k_: sg((q.get("crises") or {}).get(k_))  # noqa: E731

    def brow(share: int, q: dict, t: dict, dil: dict | None, cls: str = "") -> str:
        lab_ = "0% = Defensive launch" if share == 0 else "100% = Growth" if share == 100 else f"{share}% Growth / {100 - share}% Defensive"
        cells = [lt(lab_), td(pct(q["cagr"])), td(f2(q["xs"])), td(pct(q["vol"])), td(sg(q["dd"])), td(pct(t.get("p10"))), td(pct(t.get("p15"))), td(pct(t.get("p20"))),
                 td(crz(q, "gfc")), td(crz(q, "bear_2022"))]
        if dil:
            cells += [lt(f'Growth + {dil["cash"] * 100:.0f}% BIL'), td(pct(dil["q"]["vol"])), td(pct(dil["q"]["cagr"])), td(f2(dil["q"]["xs"])), td(sg(dil["q"]["dd"]))]
        else:
            cells += [td("–")] * 5
        return tr(cells, cls)

    bl_pts = sorted((int(nm.split()[1]), v) for nm, v in BL.items() if "diluted" in v)
    rows_b = []
    if "defensive launch" in BL:
        rows_b.append(brow(0, BL["defensive launch"]["q"], BL["defensive launch"]["tails"], None, "dim"))
    rows_b += [brow(sh_, v["q"], v["tails"], v["diluted"]) for sh_, v in bl_pts]
    rows_b.append(brow(100, g1["q"], g1["tails"], None, "pick"))
    ends_xs = [g1["q"]["xs"]] + ([BL["defensive launch"]["q"]["xs"]] if "defensive launch" in BL else [])
    mid_xs = [v["q"]["xs"] for _, v in bl_pts]
    n_xs_above = sum(x > max(ends_xs) for x in mid_xs)
    d_cagr = [(v["q"]["cagr"] - v["diluted"]["q"]["cagr"]) * 100 for _, v in bl_pts]
    d_dd = [(v["q"]["dd"] - v["diluted"]["q"]["dd"]) * 100 for _, v in bl_pts]
    d_sx = [v["q"]["xs"] - v["diluted"]["q"]["xs"] for _, v in bl_pts]
    rng = lambda xs_, d=1: f"{min(xs_):+.{d}f} … {max(xs_):+.{d}f}".replace("-", "−")  # noqa: E731
    rng_abs = lambda xs_: f"{min(0.5, min(abs(x) for x in xs_)):.1f} … {max(abs(x) for x in xs_):.1f}"  # lower end 0.5: the volatility-adjusted gap of the reviewers  # noqa: E731
    nb_ = len(bl_pts)
    dd_line = " · ".join(f"{sh_}%: {sg(v['q']['dd'])}" for sh_, v in bl_pts)
    out["BLEND"] = (
        "<h2>החוגה בין הגנתי לצמיחה</h2>"
        + points([
            f"<b>שני מוצרי בסיס וחוגה אחת:</b> ההשקה ההגנתית ו־{L('Growth')}. לקוח שרוצה משהו באמצע לא צריך מוצר שלישי; הוא מחלק את הכסף בין השניים.",
            f"<b>איך בוחרים נקודה:</b> לפי הירידה שהלקוח יכול לחיות איתה. {R}{L('Max DD')} לפי חלק ה־{L('Growth')}: {L(dd_line)}. העמודות {L('DD beyond')} מראות כמה מהמסלולים בבוטסטרפ ירדו עמוק יותר.",
            f"<b>השילוב משפר מעט את ה־{L('Sharpe')}:</b> ב־{L(str(n_xs_above))} מתוך {L(str(nb_))} נקודות הביניים ה־{L('Excess Sharpe')} גבוה משני הקצוות ({L(f'{f2(min(mid_xs))} … {f2(max(mid_xs))}')} מול {L(' / '.join(f2(x) for x in ends_xs))}).",
            f"<b>מול מזומן:</b> {L('Growth')} מדולל ב־{L('BIL')} לתנודתיות דומה נותן כמעט אותה תשואה (השילוב פחות המדולל: {L('CAGR ' + rng(d_cagr, 2) + ' pp')}) ו־{L('Sharpe')} דומה ({L(rng(d_sx, 2))}).",
            f"<b>מה ההגנתי כן מוסיף:</b> ירידה היסטורית רדודה יותר בכ־{L(rng_abs(d_dd) + ' pp')}, במסלול ההיסטורי האחד. חלק מזה בא מכך שלשורות המדוללות יש תנודתיות מעט גבוהה יותר (ראה עמודת {L('Vol')}). "
            + (f"בנקודת {L('20%')} הדילול במזומן מעט טוב יותר." if d_cagr and d_cagr[0] < 0 and d_sx[0] < 0 else ""),
            f"<b>זה לא פיזור מלא:</b> המתאם של {L('Growth')} עם ההשקה ההגנתית הוא {L(f2(cg['all']))} בכל הימים ו־{L(f2(cg['spx_worst5']))} ב־5% הימים הגרועים של {L('S&amp;P 500')}. "
            f"שניהם נשענים על מנוע ה־{L('TAA')}: {L('BTAL_QQQ')} שבהגנתי הוא אותו מנוע.",
        ])
        + table(["Growth share (the rest in the Defensive launch)", "CAGR", "Excess Sharpe", "Vol", "Max DD", "DD beyond −10%", "DD beyond −15%", "DD beyond −20%", "GFC window", "2022 bear window",
                 "About the same volatility: Growth diluted with BIL", "Vol", "CAGR", "Excess Sharpe", "Max DD"], rows_b)
        + notes([
            f"<b>איך זה חושב:</b> שני המוצרים מוחזקים במשקל קבוע ומתאפסים פעם בשנה, כמו הפודים בתוך מוצר. הכול בתוך המדגם, ב־{L('backtest')}.",
            f"<b>העמודות הימניות:</b> {L('Growth')} עם מזומן ({L('BIL')}) בכמות שנותנת בערך אותה תנודתיות כמו השילוב (ההתאמה מקורבת). זו ההשוואה: האם ההגנתי שווה יותר ממזומן.",
            f"<b>ההשקה ההגנתית:</b> {L('CORE5 54 / BTAL_QQQ 36 / cash 10')}, מחושבת כאן בשיטת 252 הימים של פרק הצמיחה.",
        ]))

    # ── the dial ──
    wc = B["edge_decay"]["worst_case"]
    step1, step2 = g2["q"]["cagr"] - g1["q"]["cagr"], g3["q"]["cagr"] - g2["q"]["cagr"]
    c1_, c2_, c3_ = cons("GR1").get("cagr"), cons("GR2").get("cagr"), cons("GR3").get("cagr")
    out["DIAL_INTRO"] = points([
        f"<b>כל צעד בסולם המוצרים קונה תשואה במחיר ריכוז.</b> הטבלה מראה את כל נקודות חוגת ה־{L('TAA')} שהוגדרו מראש; שלושת המוצרים מסומנים.",
        f"<b>צעד ראשון, {L('Growth')} ← {L('Growth Plus')}:</b> רק החלפת {L('TAA 3x')} ב־{L('TAA 3x 1N')}, באותם משקלים ({L('40 / 30 / 30')}). {R}{L(f'CAGR {sg(step1)}')} לשנה, "
        f"{L('Max DD')} מ־{L(sg(g1['q']['dd']))} ל־{L(sg(g2['q']['dd']))}, {L('Excess Sharpe')} מ־{L(f2(g1['q']['xs']))} ל־{L(f2(g2['q']['xs']))}."
        + (f" מול {L('Growth')} יש ל־{L('Growth Plus')} {L('Sharpe')} גבוה יותר רק ב־{L(pct(v21['share_xs'], 0))} מהמסלולים: הצעד קונה תשואה, לא {L('Sharpe')}." if v21 else ""),
        f"<b>צעד שני, {L('Growth Plus')} ← {L('Aggressive')}:</b> {L('TAA 3x 1N')} עולה מ־{L('40%')} ל־{L('60%')}. {R}{L(f'CAGR {sg(step2)}')} לשנה, {L('Max DD')} {L(sg(g3['q']['dd']))}, {L('Excess Sharpe')} {L(f2(g3['q']['xs']))}."
        + ("" if g3_off else f" זה פחות מ־{L('1.5 pp')} שכלל התפריט דורש, ולכן {L('Aggressive')} לא מוצע."),
        (f"<b>בתרחיש השמרני</b> (שלושה רבעים מהיתרון, עלויות המודל) הצעדים הם {L(sg(c2_ - c1_))} ו־{L(sg(c3_ - c2_))}." if None not in (c1_, c2_, c3_) else ""),
        f"<b>אם ה־{L('TAA')} מפסיק לעבוד:</b> ה־{L('Excess Sharpe')} יורד ל־{L(f2(dead('GR1')))} ב־{L('Growth')}, ל־{L(f2(dead('GR2')))} ב־{L('Growth Plus')} ול־{L(f2(dead('GR3')))} ב־{L('Aggressive')}.",
    ])
    rows = []

    prod_of = {"dial taa3x 40": "GR1", "dial taa3x_1n 40": "GR2", "dial taa3x_1n 60": "GR3"}
    for name, meta in S["dial"].items():
        b = K[name]
        dead_ = B["edge_decay"]["scenarios"]["TAA dead"].get(name)
        rows.append(tr([lt(f'{"TAA 3x" if meta["taa"] == "taa3x" else "TAA 3x 1N"} {meta["share"] * 100:.0f}%' + (f' = {GR[prod_of[name]][0]}' if name in prod_of else " = equal capital (registered default)" if name == "dial taa3x 33" else "")),
                        td(pct(b["q"]["cagr"])), td(f2(b["q"]["xs"])), td(pct(b["q"]["vol"])), td(sg(b["q"]["dd"])),
                        td(pct(b["tails"]["p20"])), td(pct(b["tails"]["p25"])),
                        td(pct(b["tails"]["p30"])), td(pct(b["frames"]["s3_plus_5bps"]["cagr"])), td(f2(dead_["xs"]) if dead_ else "–"),
                        td(f'{XB[name]["nasdaq_lookthrough"]["max"]:.2f}×'), td(capstr(name)), lt(b["strictest_rung"] or "none")], "pick" if name in prod_of else ""))
    for name, lab_ in (("capsule TAA 3x", "TAA 3x 100% (ceiling)"), ("capsule TAA 3x 1N", "TAA 3x 1N 100% (ceiling)")):
        b = K[name]
        rows.append(tr([lt(lab_), td(pct(b["q"]["cagr"])), td(f2(b["q"]["xs"])), td(pct(b["q"]["vol"])), td(sg(b["q"]["dd"])),
                        td(pct(b["tails"]["p20"])), td(pct(b["tails"]["p25"])),
                        td(pct(b["tails"]["p30"])), td(pct(b["frames"]["s3_plus_5bps"]["cagr"])), td("–"), td("up to 3.00×"), td(capstr(name.replace("capsule", "leg"))), lt(b["strictest_rung"] or "none")], "dim"))
    out["DIAL"] = table(["TAA share (satellites split the rest equally)", "CAGR", "Excess Sharpe", "Vol", "Max DD", "DD beyond −20%",
                         "DD beyond −25%", "DD beyond −30%", "CAGR +5 bps", "Excess Sharpe, TAA dead", "Nasdaq peak", "Capacity (worked route)", "Rung passed"], rows)
    rows = []
    for key, m in S["margin"].items():
        base, tgt = key.split(" -> ")
        tg = m["target"]
        hard = tg["breach_key"]
        rows.append(f'<tr class="grp"><td colspan="16">{GR[base][0]} on margin against {GR[tgt][0]} (breach limit −{hard[1:]}%)</td></tr>')
        extra = lambda n: [td(sg(K[n]["q"]["worst_year"])), td(f'{sg(K[n]["q"]["crises"]["gfc"])} / {sg(K[n]["q"]["crises"]["bear_2022"])}'), td(f2(dead(n))),  # noqa: E731
                           td(f'{XB[n]["nasdaq_lookthrough"]["max"]:.2f}×')]
        rows.append(tr([lt(f"{GR[tgt][0]} (no margin)"), td("1.00×"), td(pct(tg["q"]["cagr"])), td(pct(tg["q"]["vol"])), td(f2(tg["q"]["xs"])), td(sg(tg["q"]["dd"])), td(pct(tg["tails"][hard])),
                        td(pct(tg["plus5"]["cagr"])), td(pct(tg["exact"]["cagr"])), td("–"), td("1.00×"), td(pct(tg["reg_t"], 0))] + extra(tgt)))
        for kind, lab_ in (("vol_matched", "same volatility"), ("cagr_matched", "same CAGR"), (LEV_KEY, "fixed 1.40×, the alternative")):
            r = m.get(kind)
            if not r or r.get("name") not in K or r.get("name") not in XB:
                continue
            rows.append(tr([lt(f"{GR[base][0]} × {r['L']:.2f} ({lab_})"), td(f"{r['L']:.2f}×"), td(pct(r["q"]["cagr"])), td(pct(r["q"]["vol"])), td(f2(r["q"]["xs"])), td(sg(r["q"]["dd"])),
                            td(pct(r["tails"][hard])), td(pct(r["plus5"]["cagr"])), td(pct(r["exact"]["cagr"])), td(f'{pct(r["spread_050"]["cagr"])} / {pct(r["spread_250"]["cagr"])}'),
                            td(f'{r["peak_leverage"]:.2f}×'), td(pct(r["reg_t"], 0) + (" ⚠" if r["needs_portfolio_margin"] else ""))] + extra(r["name"])))
    m12, m13 = S["margin"]["GR1 -> GR2"]["vol_matched"], S["margin"]["GR1 -> GR3"]["vol_matched"]
    m23 = (S["margin"].get("GR2 -> GR3") or {}).get("vol_matched")
    xl = XB[m12["name"]]
    bk2, bk3 = S["margin"]["GR1 -> GR2"]["target"]["breach_key"], S["margin"]["GR1 -> GR3"]["target"]["breach_key"]

    def lev_verdict(m: dict, tgt: dict, hard: str) -> str:
        """Levered book against the dial product at the pre-registered tolerance (0.3 pp CAGR, 0.5 pp drawdown or breach)."""
        dc, dd_, db = m["q"]["cagr"] - tgt["q"]["cagr"], m["q"]["dd"] - tgt["q"]["dd"], tgt["tails"][hard] - m["tails"][hard]
        better, worse = (dc > 0.003, dd_ > 0.005, db > 0.005), (dc < -0.003, dd_ < -0.005, db < -0.005)
        if any(better) and not any(worse):
            return "המינוף טוב יותר מעבר לסף שנקבע מראש"
        if any(worse) and not any(better):
            return "המוצר שעל חוגת ה־TAA טוב יותר מעבר לסף שנקבע מראש"
        return "התמונה מעורבת" if any(better) else "תיקו בתוך הסף שנקבע מראש"

    out["MARGIN"] = (points([
        f"<b>הרעיון:</b> מינוף על {L('Growth')} שומר על האיזון בין המנועים, במקום לרכז את הסיכון ב־{L('TAA')}.",
        f"<b>מול {L('Growth Plus')}, באותה תנודתיות ({L(f'×{m12['L']:.2f}')}):</b> {L(f'CAGR {pct(m12['q']['cagr'], 2)}, Max DD {sg(m12['q']['dd'])}, DD beyond −{bk2[1:]}% {pct(m12['tails'][bk2])}')} מול "
        f"{L(f'{pct(g2['q']['cagr'], 2)}, {sg(g2['q']['dd'])}, {pct(g2['tails'][bk2])}')}: {lev_verdict(m12, g2, bk2)} ({L('0.3 pp')} תשואה, {L('0.5 pp')} ירידה או חריגה).",
        f"<b>אותה השוואה בעלות גבוהה יותר:</b> ב־{L('+5 bps')}: {L(pct(m12['plus5']['cagr'], 2))} מול {L(pct(g2['frames']['s3_plus_5bps']['cagr'], 2))}. במרווח מימון של {L('2.5%')}: {L(pct(m12['spread_250']['cagr'], 2))}. "
        f"חשיפת השיא לנאסד״ק: {L(f'{xl['nasdaq_lookthrough']['max']:.2f}×')} במינוף מול {L(f'{XB['GR2']['nasdaq_lookthrough']['max']:.2f}×')} ב־{L('Growth Plus')}.",
        f"<b>מול {L('Aggressive')} ({L(f'×{m13['L']:.2f}')}):</b> {L(f'CAGR {pct(m13['q']['cagr'], 2)}, Max DD {sg(m13['q']['dd'])}')} מול {L(f'{pct(g3['q']['cagr'], 2)}, {sg(g3['q']['dd'])}')}: {lev_verdict(m13, g3, bk3)}. "
        f"ב־{L('+5 bps')}: {L(pct(m13['plus5']['cagr'], 2))} מול {L(pct(g3['frames']['s3_plus_5bps']['cagr'], 2))}. המינוף ממנף גם את עלויות ה־{L('MR')}.",
        (f"<b>{L('Growth Plus')} במינוף מול {L('Aggressive')} ({L(f'×{m23['L']:.2f}')}):</b> {L(f'CAGR {pct(m23['q']['cagr'], 2)}, Max DD {sg(m23['q']['dd'])}')} מול {L(f'{pct(g3['q']['cagr'], 2)}, {sg(g3['q']['dd'])}')}: {lev_verdict(m23, g3, bk3)}." if m23 else ""),
        (f"<b>החלופה, {L('Growth × 1.40')} מול {L('Aggressive')}:</b> {L(f'CAGR {pct(LEV['q']['cagr'], 2)}, Max DD {sg(LEV['q']['dd'])}, DD beyond −30% {pct(LEV['tails'].get('p30'))}')} מול "
         f"{L(f'{pct(g3['q']['cagr'], 2)}, {sg(g3['q']['dd'])}, {pct(g3['tails'].get('p30'))}')}: {lev_verdict(LEV, g3, bk3)}." if LEV else ""),
        (f"<b>אותה השוואה, בשאר המדדים:</b> ב־{L('+5 bps')} {L(pct((LEV.get('plus5') or {}).get('cagr'), 2))} מול {L(pct(g3['frames']['s3_plus_5bps']['cagr'], 2))}; "
         f"חלון 2022 {L(sg(((LEV['q'].get('crises') or K.get(LEV.get('name'), {}).get('q', {}).get('crises') or {}).get('bear_2022'))))} מול {L(sg(g3['q']['crises']['bear_2022']))}; "
         f"עם {L('TAA')} מת {L(f2(dead(LEV.get('name'))))} מול {L(f2(dead('GR3')))}; שיא חשיפה לנאסד״ק {L(f'{(XB.get(LEV.get('name')) or {}).get('nasdaq_lookthrough', {}).get('max', float('nan')):.2f}×')} מול {L(f'{XB['GR3']['nasdaq_lookthrough']['max']:.2f}×')}." if LEV else ""),
        "<b>אין כאן מנצח מוכרז:</b> המינוף הוא חלופה לשלב הקרן, לא מוצר בתפריט. זו החלטה של שיקול דעת.",
        f"<b>זמינות:</b> היום כל פוד בחשבון {L('Reg-T')} נפרד, ולכן רק חוגת ה־{L('TAA')} מעשית. בקרן עם חשבון מרג׳ין אחד שתי הדרכים אפשריות.",
        f"<b>המודל:</b> הלוואה ב־{L('DTB3 + 1.5%')}, מתאפסת פעם בשנה עם התיק. אין בו קריאות מרג׳ין ואין יום פער. התאמת התנודתיות מקורבת (המינוף הממוצע בפועל {L(f'{m12['mean_leverage']:.2f}×')}). "
        f"המימון שמרני, כי בחשבון אחד ההלוואה מתקזזת מול ה־{L('BIL')} שהתיק מחזיק. כל פקודה גדולה פי {L(f'{m12['L']:.2f}')}, כך שהגודל המרבי קטן באותו יחס.",
    ])
                     + table(["Route", "Leverage", "CAGR", "Vol", "Excess Sharpe", "Max DD", "DD beyond the limit", "CAGR +5 bps", "2012+ CAGR", "CAGR at spread 0.5% / 2.5%",
                              "Peak leverage in a year", "Reg-T initial", "Worst year", "2008 / 2022", "Excess Sharpe, TAA dead", "Nasdaq peak"], rows)
                     + f'<p class="cap">חלוקת הסיכון בין המנועים במסלול המינוף זהה לזו של המוצר הממונף: המינוף מגדיל את כולם באותו יחס.</p>')

    # ── robustness ──
    ED = B["edge_decay"]
    mm = ED["minimax"]
    out["ROB_INTRO"] = points([
        "<b>השאלה:</b> כל מנוע נבחר על אותה היסטוריה, ולכן חשוב מה קורה כשאחד מהם מאבד את היתרון.",
        f"<b>״מנוע מת״:</b> הורדתי מהמנוע את כל התשואה העודפת שלו והשארתי את התנודתיות. זה קשה יותר מ״מרוויח כמו {L('T-bills')}״.",
        f"<b>עם {L('TAA')} מת:</b> {L('Growth')} נשאר עם {L(f'Excess Sharpe {f2(d_g1)}')}, {L('Growth Plus')} עם {L(f2(dead('GR2')))} ו־{L('Aggressive')} עם {L(f2(dead('GR3')))}.",
        f"<b>התיק החודשי:</b> {L('Monthly')} נשאר עם {L(f2(d_s9))}. יש בו מנוע תשואה אחד, ו־" + L("CORE5") + " לבדו מחזיק את התיק."
        + (" זה היתרון המעשי של פיזור בין מנועים." if d_s9 is not None and d_g1 > d_s9 + 0.05 else ""),
    ])
    sc_list = ["all at 0.75", "all at 0.5", "TAA dead", "Defense First dead (TAA and BTAL_QQQ)", "MOM dead", "MR dead", "TAA dead, others 0.75", "common shock (half the QQQ premium)"]
    rows = []
    has_s13 = S13 in ED["scenarios"]["TAA dead"]
    for n, he, cls in ((fin["GR1"], GR["GR1"][1], "pick"), (fin["GR2"], GR["GR2"][1], ""), (fin["GR3"], GR["GR3"][1], ""), (S9, "החודשי", ""),
                       (S13, "החודשי מ־1 באוקטובר, ארבעה פודים (להשוואה)", "dim"), ("S4 core + satellites", "", ""), ("S11 cluster parity", "", ""), ("S1 no momentum", "", "")):
        if n not in K or not ED["scenarios"]["TAA dead"].get(n):
            continue
        cells = [name_cell(he or show(n)), td(f'{pct(K[n]["q"]["cagr"])} / {f2(K[n]["q"]["xs"])}')]
        for sc in sc_list:
            r = ED["scenarios"][sc].get(n)
            cells.append(td(f'{pct(r["cagr"])} / {f2(r["xs"])}' if r else "–"))
        w = wc.get(n)
        cells.append(td(f'{f2(w["xs"])} ({w["scenario"].split()[0]}); no reset {f2(w["no_reset_xs"])}' if w else "–"))
        dd_ = ED["scenarios"]["TAA dead"].get(n)
        cells.append(td(pct(dd_["p20"]) if dd_ else "–"))
        rows.append(tr(cells, cls))
    mm_txt = (f"{L('Growth')} במקרה הגרוע (מנוע אחד מת): {L(f2(mm['gr1_worst_xs']))}. הטוב ביותר בין השכנים והמתחרים: {L(f2(mm['best_rival_worst_xs']))} ({L(_html.escape(nb_name(mm['best_rival'])))}). "
              + ("ההפרש בתוך 0.02, ולכן מותר לומר שהמבנה הזה גם ממזער את המקרה הגרוע." if mm["gr1_is_minimax_within_0.02"]
                 else f"לכן אני לא טוען שהמבנה הזה הוא ״מינימקס״. תיקים עם פחות משקל ל־{L('TAA')}, כולל ההון השווה, מחזיקים טוב יותר כשהמנוע הזה מת. זה המחיר של ההטיה."))
    beta_taa = ED["beta_qqq"]["taa3x"]
    c_s0 = cons(S0)
    out["DECAY"] = (table(["Book: CAGR / Excess Sharpe", "Backtest", "All engines at 3/4 of the edge", "All at 1/2", "TAA dead", "TAA and BTAL_QQQ dead", "MOM dead", "MR dead",
                           "TAA dead, others 3/4", "Regression-beta shock (half the QQQ premium × weekly beta)", "Worst single-engine Excess Sharpe", "DD beyond −20%, TAA dead"], rows)
                    + notes([
                        f"<b>לא ״מינימקס״:</b> {mm_txt}" if not mm["gr1_is_minimax_within_0.02"] else f"<b>המקרה הגרוע:</b> {mm_txt}",
                        f"<b>״{L('TAA dead')}״ בתיקים החודשיים:</b> בתיק החודשי יש פוד {L('TAA')} אחד ואין {L('BTAL_QQQ')}, ולכן העמודה ״{L('TAA and BTAL_QQQ dead')}״ זהה אצלו ל״{L('TAA dead')}״. "
                        + (f"בתיק הישן מ־1 באוקטובר (שורת ההשוואה) {L('BTAL_QQQ')} רץ על אותו מנוע, ולכן שם העמודה הזאת מכבה את שניהם." if has_s13 else ""),
                        f"<b>{L('No reset')}:</b> אותו תרחיש בלי האיפוס השנתי. האיפוס מעביר כל שנה כסף למנוע המת. ירידת השיא בתרחישים האלה לא מפורשת.",
                        f"<b>עמודת ה־{L('shock')} מקלה:</b> הבטא של הרגרסיה נמוכה מהחשיפה הממוצעת בפועל (ב־{L('TAA 3x')}: {L(f2(beta_taa))} מול כ־{L('0.69')} לפי חישוב הסוקר), כי הפודים יוצאים מהשוק כשהתנודתיות גבוהה.",
                        f"<b>חישוב של הסוקר הבלתי תלוי:</b> עם החשיפה הממוצעת בפועל, חצי מפרמיית הנאסד״ק נותן {L('equal-capital Growth 11.9% / 0.85')} (חושב על גרסת ההון השווה ולא חושב מחדש).",
                        (f"<b>להשוואה:</b> העמודה ״{L('3/4 of the edge')}״ (התרחיש השמרני) נותנת להון השווה {L(f'{pct(c_s0['cagr'])} / {f2(c_s0['xs'])}')}. "
                           + ("כלומר התרחיש של הסוקר חמור יותר מהתרחיש השמרני." if c_s0["cagr"] > 0.119 + 0.005 else "כלומר הוא קרוב לתרחיש השמרני." if abs(c_s0["cagr"] - 0.119) <= 0.005 else "כלומר התרחיש השמרני חמור יותר.") if c_s0 else ""),
                    ]))
    rows = []
    slot_he = {"T1 MOM -> BIL": f"המומנטום מוחלף ב־{L('BIL')}", GR1_L: f"ה־{L('MR')} מוחלף ב־{L('BIL')}", "T3 TAA -> BIL": f"ה־{L('TAA')} מוחלף ב־{L('BIL')}",
               "T4 MOM -> QQQ": f"המומנטום מוחלף ב־{L('QQQ')}"}
    for t_, r in S["slots"].items():
        f_ = r["frames"]
        rows.append(tr([name_cell(slot_he[t_]), td(pct(r["q"]["cagr"])), td(f2(r["q"]["xs"])), td(sg(r["q"]["dd"])), td(pct(r["tails"]["p20"])),
                        td(f'{pct(f_["main"]["share_xs"], 0)} / {pct(f_["plus5"]["share_xs"], 0)} / {pct(f_["plus10"]["share_xs"], 0)}'),
                        td(f'{pct(f_["main"]["share_cagr"], 0)} / {pct(f_["plus10"]["share_cagr"], 0)}'),
                        td(" / ".join(pct(r["blocks"][k]["share_xs"], 0) for k in ("A", "B", "C")))]))
    rows.insert(0, tr([name_cell(f"{R}{L('Growth')} כמו שהוא"), td(pct(g1["q"]["cagr"])), td(f2(g1["q"]["xs"])), td(sg(g1["q"]["dd"])), td(pct(g1["tails"]["p20"])), td("–"), td("–"), td("–")], "pick"))
    t4 = K["T4 MOM -> QQQ"]
    t4_rung = t4.get("strictest_rung")
    mom_bil = 1 - t1["main"]["share_xs"]
    out["SLOTS"] = (table(["Growth with one capsule replaced", "CAGR", "Excess Sharpe", "Max DD", "DD beyond −20%", "Growth has the higher Excess Sharpe: paths at 0 / +5 / +10 bps",
                           "Higher CAGR: paths at 0 / +10 bps", "Higher Excess Sharpe by block A / B / C"], rows)
                    + notes([
                        f"<b>ה־{L('MR')} מרוויחה את המקום:</b> {L('Growth')} גבוה ב־{L('Sharpe')} מהגרסה עם {L('BIL')} במקומה ב־{L(pct(t2['main']['share_xs'], 0))} מהמסלולים, וב־{L('+5 bps')} ב־{L(pct(t2['plus5']['share_xs'], 0))} "
                        f"({bar(t2['plus5']['share_xs'])} {L('80%')}). נקודת האיזון מול {L('BIL')}: כ־{L(f'{gate['xs']['breakeven_bps_per_side']:.0f} bps')} לצד ב־{L('Sharpe')} וכ־{L(f'{gate['cagr']['breakeven_bps_per_side']:.0f} bps')} ב־{L('CAGR')}.",
                        f"<b>המומנטום מוסיף תשואה, לא {L('Sharpe')}:</b> עם {L('BIL')} במקומו ה־{L('Sharpe')} גבוה יותר ב־{L(pct(mom_bil, 0))} מהמסלולים ({bar(mom_bil)} {L('80%')}), והתשואה נמוכה ב־{L(pct(g1['q']['cagr'] - K['T1 MOM -> BIL']['q']['cagr']))} לשנה. "
                        f"הוא בתיק בשביל תשואה בתוך תקרת הסיכון.",
                        f"<b>{L('QQQ')} במקום המומנטום:</b> {L(f'CAGR {pct(t4['q']['cagr'])}, Max DD {sg(t4['q']['dd'])}')}. " + ("יותר תשואה, אבל זה שובר את מדרגת " + L("GROWTH") + "." if t4_rung != "GROWTH" and t4["q"]["cagr"] > g1["q"]["cagr"] else ""),
                        f"<b>איך לקרוא:</b> עמודת {L('+10 bps')} היא אקסטרפולציה ליניארית מ־{L('+5 bps')}. העמודה לפי בלוק מחושבת על 6,000 מסלולים.",
                    ]))

    ck = ["c0_rung", "c1_share_ge_80", "c2_breach_no_worse", "c3_exact_xs_higher", "c4_plus5_xs_not_lower", "c5_h1_higher", "c6_h2_higher"]
    rows = []
    for c in S["challenges"]:
        b = K[c["challenger"]]
        rows.append(tr([lt(show(c["challenger"])), lt(mix(b["weights"]) if b["weights"] else "weights from trailing volatility"), td(pct(b["q"]["cagr"])), td(f2(b["q"]["xs"])), td(sg(b["q"]["dd"])),
                        td(pct(b["tails"]["p20"])), td(pct(c["share_xs"], 0) + (" ~" if c["borderline"] else "")),
                        td(f'{c["gap_xs_p5_50_95"][0]:+.2f} … {c["gap_xs_p5_50_95"][2]:+.2f}'), td(f'{c["xs_monthly"][0] - c["xs_monthly"][1]:+.2f}'), td(pct(c["share_cagr"], 0))]
                       + [td(("exempt (" + ("passes" if c["rung_pass"] else "fails") + ")") if c["challenger"] == "S10 old G3" else yes(c["rung_pass"] if c["challenger"] == S9 else c["checks"]["c0_rung"]))]
                       + [td(yes(c["checks"][k])) for k in ck[1:]] + [td(yes(c["passed"])), td(yes(c["passed_without_c2"]))]
                       + [td(" / ".join(f'{c["block_gap_xs"][k_]:+.2f}' for k_ in ("A", "B", "C")))]
                       + [td("flips" if (c["xs"][0] - c["xs"][1]) * (c["xs_monthly"][0] - c["xs_monthly"][1]) < 0 else "")]))
    passed = [c["challenger"] for c in S["challenges"] if c["passed"]]
    cstr = S["construction"]
    rows2 = [tr([lt(show(n)), td(pct(v["cagr"])), td(f2(v["xs"])), td(sg(v["dd"])), td(pct(v["p20"]))], "pick" if n == "GR1" else "") for n, v in cstr["rows"].items()]
    rows3 = []
    for n, v in S["standins"].items():
        b = K[n]
        rows3.append(tr([lt(show(n)), td(pct(b["q"]["cagr"])), td(f2(b["q"]["xs"])), td(sg(b["q"]["dd"])), td(pct(b["tails"]["p20"])), td(pct(b["frames"]["s3_plus_5bps"]["cagr"])),
                         td(yes(v["rung"]["pass"])), td(pct(v["vs_s9"]["share_xs"], 0)), td(yes(v["vs_s9"]["passed"])), td(f2(v["corr_with_gr1"]))]))
    o2 = S["gr2_vs_old_plus"]
    chd = {c["challenger"]: c for c in S["challenges"]}
    ck_he = {"c0_rung": "המדרגה", "c1_share_ge_80": f"רף ה־{L('80%')}", "c2_breach_no_worse": L("breach"), "c3_exact_xs_higher": f"חלון {L('2012+')}", "c4_plus5_xs_not_lower": L("+5 bps"),
             "c5_h1_higher": "החצי הראשון", "c6_h2_higher": "החצי השני"}
    fails = lambda c: ", ".join(ck_he[k_] for k_ in ck if not c["checks"][k_]) or "אין"  # noqa: E731
    s8, s5, s1c = chd["S8 GR1 75 / DEF 25"], chd["S5 MR tilt"], chd["S1 no momentum"]
    s0c = chd.get(S0)
    out["CHALLENGERS"] = (
        points([
            f"<b>התוצאה:</b> {L(str(len(S['challenges'])))} מתחרים נבדקו מול {L('Growth')} בשבע בדיקות. " + (f"עברו: {L(', '.join(show(x) for x in passed))}." if passed else "אף אחד לא עבר את כולן.")
            + f" זה לא אישור ש־{L('40 / 30 / 30')} הוא הטוב ביותר, רק שאין ראיה נגדו. המבחן חלש: מתחרה שבאמת טוב ב־{L('0.05')} {L('Sharpe')} עובר בפחות מחצי מהמקרים.",
            f"<b>{L('Growth')} עם רבע בליבה ההגנתית ({L('75 / 25')}):</b> {L('Sharpe')} גבוה יותר ב־{L(pct(s8['share_xs'], 0))} מהמסלולים. נכשל ב: {fails(s8)}. המחיר שלו: {L(f'{(g1['q']['cagr'] - K['S8 GR1 75 / DEF 25']['q']['cagr']) * 100:.1f} pp')} פחות תשואה.",
            f"<b>יותר {L('MR')}, פחות מומנטום ({L('50 / 15 / 35')}):</b> גבוה ב־{L(pct(s5['share_xs'], 0))} מהמסלולים. נכשל ב: {fails(s5)}.",
            f"<b>בלי מומנטום ({L('TAA 3x 50 / MR 50')}):</b> גבוה ב־{L(pct(s1c['share_xs'], 0))} מהמסלולים. נכשל ב: {fails(s1c)}.",
            (f"<b>ההון השווה (ברירת המחדל הרשומה):</b> " + ("עובר" if s0c["passed"] else "לא עובר") + f" מול {L('40 / 30 / 30')} ({L(pct(s0c['share_xs'], 0))} מהמסלולים). שני תיקים כמעט זהים." if s0c else ""),
            f"<b>מה עושים עם זה:</b> לפי הכלל שנקבע מראש שום מתחרה לא מחליף מוצר במחקר הזה. הקרובים ייעקבו בנייר ליד {L('Growth')}.",
            f"<b>בכיוון ההפוך, בעלויות המודל:</b> {L('Growth')} " + ("עובר את כל שבע הבדיקות" if rev["passed"] else f"לא עובר את כל הבדיקות (נכשל ב: {fails(rev)})")
            + f" מול {L('Monthly')} ({L(pct(rev['share_xs'], 0))} מהמסלולים ב־{L('Sharpe')}, {L(pct(rev['share_cagr'], 0))} ב־{L('CAGR')}). "
            + "ראה הטבלה ״מול התיק החודשי״.",
        ])
        + table(["Challenger against Growth", "Capital mix", "CAGR", "Excess Sharpe", "Max DD", "DD beyond −20%", "Higher Excess Sharpe: paths", "Sharpe gap, 90% interval", "Monthly-Sharpe gap", "Higher CAGR: paths", "0 rung", "1 ≥80%",
                 "2 breach", "3 2012+", "4 +5 bps", "5 H1", "6 H2", "Passed", "Passed without check 2", "Sharpe gap by block A / B / C", "Order on monthly returns"], rows)
        + notes([
            f"<b>~</b> = גבולי ({L('75%')} עד {L('85%')}).",
            "<b>בדיקה 2</b> הוכרעה חלקית ממה שכבר נראה, ולכן מוצגת גם התוצאה בלעדיה.",
            f"<b>{L('Inverse volatility, walk-forward')}:</b> משקל הפוך לתנודתיות נותן כמעט הכול למנוע שיושב במזומן (2009: {L('99%')} מומנטום; 2018: {L('96% MR')}), ולכן הוא כלל גרוע כאן.",
            f"<b>{L('flips')}</b> = הסדר מול {L('Growth')} מתהפך בין {L('Sharpe')} יומי לחודשי.",
            f"<b>שני התיקים החודשיים בטבלה:</b> החדש (שני פודים) והתיק מ־1 באוקטובר (ארבעה פודים), שנמצא כאן להשוואה.",
        ])
        + f"<h3 class=\"sub\">אותו עיקרון, כללים אחרים (דירוג {L('Growth')}: {L(f'CAGR {cstr['gr1_rank']['cagr']}, Excess Sharpe {cstr['gr1_rank']['xs']}, Max DD {cstr['gr1_rank']['dd']}, breach {cstr['gr1_rank']['p20']}')} מתוך {cstr['n']})</h3>"
        + table(["Construction", "CAGR", "Excess Sharpe", "Max DD", "DD beyond −20%"], rows2)
        + f"<h3 class=\"sub\">גרסאות עם מה שמחווט היום</h3>"
        + table(["Stand-in book", "CAGR", "Excess Sharpe", "Max DD", "DD beyond −20%", "CAGR +5 bps", "GROWTH rung", "Beats Monthly: paths", "All checks vs Monthly", "Corr with Growth"], rows3)
        + f'<p class="cap">תיק שמחזיק {L("DV2")} ו־{L("HPI")} בלי השער נשאר מאחורי אותו שער עלות של ה־{L("MR")}. כל הטבלאות בפרק הזה הן השוואות במסגרת ה־{L("backtest")} (עלויות המודל), אלא אם כתוב אחרת.</p>')

    PL, RS = B["plateau"], B["reset"]
    rows = []
    for code in GR:
        s_ = PL[code]["summary"]
        rows.append(tr([name_cell(GR[code][1])]
                       + [td(f'{pct(s_[k]["min"])} … {pct(s_[k]["median"])} … {pct(s_[k]["max"])}  (#{s_[k]["rank_from_best"]})') for k in ("cagr",)]
                       + [td(f'{f2(s_["xs"]["min"])} … {f2(s_["xs"]["median"])} … {f2(s_["xs"]["max"])}  (#{s_["xs"]["rank_from_best"]})')]
                       + [td(f'{sg(s_["dd"]["min"])} … {sg(s_["dd"]["median"])} … {sg(s_["dd"]["max"])}  (#{s_["dd"]["rank_from_best"]})')]
                       + [td(f'{pct(s_["breach"]["min"])} … {pct(s_["breach"]["median"])} … {pct(s_["breach"]["max"])}  (#{s_["breach"]["rank_from_best"]})')]
                       + [td(yes(PL[code]["rung_robust"])), lt("; ".join(PL[code]["flags"]) or "none")]))
    t_pl = table(["Product: min … median … max of 19 neighbours (rank of the product)", "CAGR", "Excess Sharpe", "Max DD", "DD beyond its limit", "Median neighbour passes the rung", "Flags"], rows)
    rows = []
    for code in GR:
        pol = RS[code]["policies"]
        sp = RS[code]["start_month_spread"]
        rows.append(tr([name_cell(GR[code][1])] + [td(f'{pct(pol[p]["free"]["cagr"])} / {f2(pol[p]["free"]["xs"])}') for p in ("annual", "none", "quarterly", "monthly")]
                       + [td(f'{pct(pol["monthly"]["charged"]["cagr"])}'), td(f'{pct(sp["cagr"]["min"])} … {pct(sp["cagr"]["max"])}'), td(yes(sp["cagr"]["january_in_middle_half"] and sp["xs"]["january_in_middle_half"])),
                          td(pct(sp["cagr"]["planning_figure"])), td(" / ".join(f'{pol["annual"]["max_capsule_share"][c] * 100:.0f}' for c in ("TAA", "MOM", "MR")))]))
    t_rs = table(["Reset policy: CAGR / Excess Sharpe", "Annual, January (the rule)", "Never", "Quarterly", "Monthly", "Monthly, 2.5 bps on capital moved", "CAGR over the 12 start months",
                  "January inside the middle half", "Median of the 12 start months (the rule's figure when January is outside)", "Largest capital share inside a year, % TAA / MOM / MR"], rows)
    rows = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")):
        sy, ro, lo = B["start_years"][n], B["rolling"][n], B["loyo"][n]
        cg = [v["cagr"] for v in sy.values()]
        xs = [v["xs"] for v in sy.values()]
        rows.append(tr([name_cell(he), td(f"{pct(min(cg))} … {pct(max(cg))}"), td(f"{f2(min(xs))} … {f2(max(xs))}"), td(f'{f2(lo["xs_min"])} … {f2(lo["xs_max"])}'),
                        td(str(lo["year_whose_removal_hurts_most"])), td(f'{f2(ro["3y"]["xs_min"])} / {f2(ro["3y"]["xs_p10"])} / {f2(ro["3y"]["xs_median"])}'),
                        td(f'{pct(ro["3y"]["cagr_min"])} / {pct(ro["3y"]["cagr_median"])}'), td(f'{pct(ro["5y"]["cagr_min"])} / {pct(ro["5y"]["cagr_median"])}')]))
    t_sy = table(["Book", "CAGR by start year 2008…2021", "Excess Sharpe by start year", "Excess Sharpe, one year left out", "Year whose removal hurts most",
                  "Rolling 3y Excess Sharpe: min / P10 / median", "Rolling 3y CAGR: min / median", "Rolling 5y CAGR: min / median"], rows)
    rsa = B["rolling_share_above_all"]
    rows_r = [tr([name_cell(GR[c][1])] + [td(" / ".join(pct(rsa[fin[c]][key][o], 0) for key in ("xs_3y", "xs_5y", "cagr_3y", "cagr_5y")))
                                          for o in ("capsule TAA 3x", "capsule TAA 3x 1N", "capsule MOM", "capsule MR", S9)]) for c in GR]
    t_ra = table(["Share of rolling windows above: Sharpe 3y / Sharpe 5y / CAGR 3y / CAGR 5y", "TAA 3x alone", "TAA 3x 1N alone", "MOM alone", "MR alone", "Monthly"], rows_r)
    out["PLATEAU"] = (points([
        f"<b>שכנים:</b> 19 תיקים שכנים לכל מוצר (עד {L('±10 pp')} לכל קפסולה).",
        f"<b>סימון ״הטוב בשכונה״:</b> מוצר שנמצא בין שלושת הטובים בשכונה שלו (נבדק על {L('Max DD')} ו־{L('Excess Sharpe')} בלבד). זה אות אזהרה (צפה לתוצאה פחות טובה), לא הישג.",
    ]) + t_pl + '<p class="cap">האיפוס נשאר שנתי בינואר; הבדיקה רק מודדת כמה מזל יש בתאריך.</p>' + t_rs + t_sy + t_ra)

    # ── dependence ──
    DP = B["dependence"]
    names = ["TAA 3x", "TAA 3x 1N", "MOM", "MR"]
    sub = lambda m: {a: {b: m[a][b] for b in names} for a in names}  # noqa: E731
    td_ = DP["tail_dependence"]
    out["DEP_INTRO"] = points([
        "<b>המתאם היומי הממוצע נמוך, אבל הוא לא קבוע.</b>",
        f"<b>כשהשער של ה־{L('MR')} סגור</b> (רוב הימים הרגועים) החלק שלו יושב ב־{L('BIL')}, והתיק הוא בעצם צמד הנאסד״ק ועוד מזומן.",
        "<b>כשהשער פתוח</b> כל השלושה מחזיקים מניות, והמתאם עולה.",
        f"<b>בירידות חדות של {L('21')} יום</b> הלוויינים נופלים יחד: {L(f'P = {pct(td_['21d']['MOM | MR'], 0)}')} מול {L('5%')} באי־תלות.",
    ])
    out["DEP_HEAT"] = (heat(sub(DP["full"]), names, "All days") + heat(sub(DP["d21"]), names, "21-session returns") + heat(sub(DP["gate_open"]), names, f"MR gate open ({DP['gate_open_share'] * 100:.0f}% of days)")
                       + heat(sub(DP["gate_closed"]), names, "MR gate closed") + heat(sub(DP["spx_worst5"]), names, "S&amp;P 500 worst 5% days") + heat(sub(DP["qqq_worst5"]), names, "QQQ worst 5% days"))
    rows = []
    for code in GR:
        d_ = dep[code]
        fl = d_["flags"]
        dr, pv = d_["diversification_ratio"], d_["vol_predicted_vs_realised"]
        rows.append(tr([name_cell(GR[code][1]), td(f'{f2(dr["full"])} / {f2(dr["h1"])} / {f2(dr["h2"])}'), td(f'{pct(pv["h2_from_h1_corr"][0])} → {pct(pv["h2_from_h1_corr"][1])}'),
                        td(f'{pct(pv["h1_from_h2_corr"][0])} → {pct(pv["h1_from_h2_corr"][1])}'), td(yes(not fl["realised_above_predicted_15pct"])), td(f2(fl["max_corr_move"]) + (" ⚠" if fl["corr_moved_more_than_0.20"] else "")),
                        lt("; ".join(f"{k} {v:.2f}" for k, v in fl["corr_above_0.70"].items()) or "none"),
                        td(f'{f2(d_["enb"]["full"])} / {f2(d_["enb"]["h1"])} / {f2(d_["enb"]["h2"])} / {f2(d_["enb"]["spx_worst5"])}'),
                        td(f'{d_["months_all_three_lost"]["count"]} of {d_["months_all_three_lost"]["of"]}; mean {sg(d_["months_all_three_lost"]["book_mean"])}, worst {sg(d_["months_all_three_lost"]["book_worst"])}'),
                        td(f'{sg(d_["at_max_dd"]["book"])}: ' + " / ".join(sg(v) for v in d_["at_max_dd"]["capsules"].values()))]))
    t_st = table(["Product", "Diversification ratio: full / H1 / H2", "H2 vol: predicted from H1 correlations → realised", "H1 vol: predicted from H2 → realised",
                  "Realised within 15% of predicted", "Largest correlation move between halves", "Pairs above 0.70", "Effective bets: full / H1 / H2 / worst days",
                  "Months when all three lost", "Deepest drawdown: book: TAA / MOM / MR"], rows)
    rows = [tr([lt(k), td(pct(td_["daily"][k], 0)), td(pct(td_["21d"][k], 0)), td(f'{f2(DP["rolling_252"][k]["min"])} / {f2(DP["rolling_252"][k]["median"])} / {f2(DP["rolling_252"][k]["max"])}')])
            for k in td_["daily"]]
    t_td = table(["Pair", "P(B in its worst 5% | A in its worst 5%), daily (5% if independent)", "same, 21-session returns", "Rolling 252d correlation: min / median / max"], rows)
    rows = [tr([lt(f'{w["start"]} → {w["end"]}'), td(sg(w["book"]), "neg")] + [td(sg(w[c]), "neg" if w[c] < 0 else "pos") for c in ("TAA 3x", "MOM", "MR")]) for w in dep["GR1"]["ten_worst_21d"]]
    t_w = table(["Growth: ten worst 21-session windows", "Growth", "TAA 3x", "MOM", "MR"], rows)
    out["DEP_NOTE"] = notes([
        f'<b>זהירות בקריאה:</b> המתאמים ומספר ה־{L("effective bets")} בימים הגרועים של השוק מוטים כלפי מטה (הימים נבחרו לפי גורם משותף), ולכן הם לא ראיה שהפיזור משתפר במשבר.',
        f'<b>המדד הנכון הוא ההסתברות לנפילה משותפת:</b> {L(" · ".join(f"{k}: {pct(td_['daily'][k], 0)}" for k in td_["daily"]))}, מול {L("5%")} באי־תלות.',
    ])
    out["DEP_MORE"] = (t_st + t_td + t_w + '<div class="two">' + heat(sub(DP["weekly"]), names, "Weekly returns")
                       + heat(sub(DP["blocks"]["A"]), names, "Block A, 2008-03 to 2012-10 (proxy era)") + "</div>")

    # ── how much to believe ──
    BV, BT = B["believe"], B["bootstrap"]
    xb_, xc_ = g1["blocks"]["B"]["xs"], g1["blocks"]["C"]["xs"]
    out["BELIEVE_INTRO"] = points([
        f"<b>מספר הכותרת הוא ה־{L('backtest')}.</b> המנוע כבר מחייב עמלות {L('IBKR')} ו־{L('2.5 bps')} החלקה לצד, ומסגרת הכותרת מזכה ריבית על מזומן פנוי. לכן הכרטיסים והטבלאות מציגים את ה־{L('backtest')} בלבד.",
        f"<b>מה שנשאר נכון:</b> אין כאן אף מספר מחוץ למדגם. כל ההיסטוריה עד אוגוסט 2026 שימשה לבחירת המנועים, והמשקלים והתיק החודשי נבחרו אחרי התוצאות. "
        f"ה־{L('Excess Sharpe')} של {L('Growth')} היה {L(f2(xb_))} ב־2012 עד 2021 ו־{L(f2(xc_))} מאז 2022.",
        f"<b>{L('Conservative case')}:</b> כל מנוע שומר שלושה רבעים מהתשואה העודפת ההיסטורית שלו, בעלויות המודל. זה התרחיש שכדאי להחזיק בראש ליד ה־{L('backtest')}.",
        f"<b>למה בלי {L('+5 bps')}:</b> בגרסה הקודמת התרחיש הזה כלל גם {L('+5 bps')} לצד. ב־{L('TAA')} ובמומנטום (פקודות חודשיות) זו ספירה כפולה של עלות שהמנוע כבר מחייב. "
        f"היא שאלה אמיתית רק בפקודות המניות היומיות של ה־{L('MR')}, וזה בדיוק מה ששער המסחר בנייר מודד.",
        f"<b>{L('Stress case')}:</b> חצי מהתשואה העודפת, ועוד {L('+5 bps')} לצד על כל דולר נסחר. זה תרחיש קיצון, לא ציפייה.",
        "<b>שניהם מוסכמות, לא תחזית.</b> אף אחד לא יודע כמה מהיתרון יישאר. טבלאות הראיות (מתחרים, מבחני מקום, מינוף, אלפא) הן השוואות במסגרת ה־" + L("backtest") + ".",
    ])
    rows = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")):
        hkb = "p20" if n in (fin["GR1"],) else hk3 if n == fin["GR3"] else HARD.get(K[n].get("strictest_rung") or "", "p20") if n == S9 else "p25"
        c_, s_, v1, v75 = cons(n), stress(n), (BV.get(n) or {}).get("k1"), (BV.get(n) or {}).get("k075")
        bc = BT["cagr"].get(n)
        two = lambda r: f'{pct(r.get("cagr"))} / {f2(r.get("xs"))}' if r else "–"  # noqa: E731
        p3 = lambda v: f'{pct(v["p_xs_below_1.0"], 0)} / {pct(v["p_xs_below_0.75"], 0)} / {pct(v["p_xs_below_0.5"], 0)}' if v else "–"  # noqa: E731
        rows.append(tr([name_cell(he, f"limit −{hkb[1:]}%"), td(f'{pct(K[n]["q"]["cagr"])} / {f2(K[n]["q"]["xs"])}'), td(sg(K[n]["q"]["dd"])),
                        td(two(c_)), td(sg(c_.get("dd"))), td(two(s_)), td(sg(s_.get("dd"))),
                        td(" / ".join(pct(x) for x in (K[n]["tails"].get(hkb), c_.get(hkb), s_.get(hkb)))),
                        td(f'{f2(v1["xs_p5_50_95"][0])} … {f2(v1["xs_p5_50_95"][2])}' if v1 else "–"),
                        td(f'{pct(v1["cagr_p5_50_95"][0])} … {pct(v1["cagr_p5_50_95"][2])}' if v1 else "–"), td(p3(v1)), td(p3(v75)),
                        td(f'{f2(BV[n]["dsr_N100"]["dsr"])} / {f2(BV[n]["dsr_N1000"]["dsr"])}' if n in BV else "–"),
                        td(f'{pct(bc["net_p5"])} / {pct(bc["net_p25"])} / {pct(bc["net_p50"])}' if bc else "–")], "pick" if n == fin["GR1"] else ""))
    out["BELIEVE"] = (table(["Book", "Backtest: CAGR / Excess Sharpe", "Backtest Max DD", "Conservative case (3/4 of the excess return, model costs): CAGR / Excess Sharpe", "Conservative case: Max DD",
                             "Stress case (1/2 of the excess return, +5 bps per side): CAGR / Excess Sharpe", "Stress case: Max DD", "DD beyond the limit: backtest / conservative / stress",
                             "Excess Sharpe, 90% bootstrap interval", "CAGR, 90% interval", "P(Excess Sharpe below 1.0 / 0.75 / 0.5)", "same at 3/4 of the excess return",
                             "Deflated Sharpe, N = 100 / 1,000 trials", "Net 2/20 CAGR: P5 / P25 / median"], rows)
                      + notes([
                          f"<b>{L('DD beyond the limit')}:</b> הגבול של כל תיק כתוב מתחת לשם שלו ({L('−20%')} למדרגת {L('GROWTH')}, {L('−25%')} למדרגת {L('GROWTH PLUS')}, {L('−30%')} למדרגת {L('AGGRESSIVE')}).",
                          f"<b>{L('Deflated Sharpe')}:</b> בודק רק אם ה־{L('Sharpe')} גדול מאפס אחרי {L('N')} ניסיונות. הוא לא אומר כמה לקצץ, והניסיונות מתואמים.",
                      ]))
    em = ED["edge_margin"]
    rows = []
    s9hk = HARD.get(K[S9].get("strictest_rung") or "", "p20")
    for n, he, hk in ((fin["GR1"], GR["GR1"][1], "p20"), (fin["GR2"], GR["GR2"][1], "p25"), (fin["GR3"], GR["GR3"][1] + (" (GROWTH PLUS, the rung it passes)" if rung3 == "GROWTH PLUS" else " (the GROWTH PLUS limit, for reference: it does not pass that rung)"), "p25"),
                      (fin["GR3"], GR["GR3"][1] + (((f" (AGGRESSIVE, the only rung it passes; approved {AGG_APPROVED})" if AGG_APPROVED else " (AGGRESSIVE, not yet confirmed)") if needs_agg else " (AGGRESSIVE)") if g3_off
                                                    else " (AGGRESSIVE rung: not decided, moot while not offered)"), "p30"), (S9, "החודשי", s9hk)):
        if n not in BT["blocks"]["63"]:
            continue
        blk = BT["blocks"]
        dk = lambda sc: ED["scenarios"][sc][n][hk]  # noqa: E731
        hz = BT["horizons"]
        rows.append(tr([name_cell(he, f"limit −{hk[1:]}%")] + [td(pct(blk[b][n][hk], 0 if blk[b][n][hk] > 0.1 else 1)) for b in ("1", "21", "63", "126", "252")]
                       + [td(pct(dk("all at 0.75"))), td(pct(dk("all at 0.5"))), td(f'{pct(hz["756"][n][hk])} / {pct(hz["1260"][n][hk])}'),
                          td(pct(BT["frames"]["s2_proxy_unscaled"][n][hk])), td(pct(BT["frames"]["s6_exact"][n][hk])),
                          td(f'{f2(BT["variance_ratio"][n]["21"])} / {f2(BT["variance_ratio"][n]["63"])}'),
                          td(("not computed" if margin_k(n) is None else f"{margin_k(n):.2f}") if n == S9 else "not computed" if n not in em else ("none" if em[n]["by_limit"][hk]["lowest_passing_k"] is None else f'{em[n]["by_limit"][hk]["lowest_passing_k"]:.2f}')),
                          td(f'{pct(K[n]["tails"][hk + "_max"])} / {pct(K[n]["tails_plus5"][hk])}')]))
    k3, k2 = em3, em2
    holds = lambda k_: "עובר" if (k_ is not None and k_ <= 0.75) else "לא עובר"  # noqa: E731
    out["RUNG"] = (points([
        f"<b>{L('DD beyond −X%')} הוא סרגל על מוסכמה אחת,</b> לא הסיכוי לירידה כזאת. הוא תלוי מאוד באורך הבלוק ובשאלה כמה מהיתרון ההיסטורי נשאר.",
        "<b>עשרת הזרעים</b> מודדים רק רעש של הסימולציה.",
        f"<b>״{L('Edge margin')}״</b> הוא החלק הקטן ביותר של היתרון שבו המוצר עדיין עובר את המדרגה שלו.",
    ])
                   + table(["DD beyond the product's limit", "Block 1 (independent days)", "Block 21", "Block 63 (the rule)", "Block 126", "Block 252", "Excess return 3/4", "Excess return 1/2",
                            "First 3 / 5 years of a path", "Unscaled proxy", "2012+ window", "Variance ratio 21 / 63", "Edge margin (lowest k that passes)", "Worst seed / at +5 bps"], rows)
                   + notes([
                       f'<b>{L("Growth")} ב־{L("−17%")}:</b> {L(pct(K[fin["GR1"]]["tails"]["p17"]))} (זרע גרוע {L(pct(K[fin["GR1"]]["tails"]["p17_max"]))}).',
                       f'<b>בשלושה רבעים מהיתרון:</b> {L("Growth Plus")} {holds(k2)} את הגבול שלו, {L("−25%")} ({L("edge margin")} {L(fk(k2))}). {R}{L("Aggressive")} {holds(k3)} את הגבול שלו, {L("−" + hk3[1:] + "%")} ({L(fk(k3))}).',
                       f'<b>התיק החודשי:</b> {monthly_margin}.',
                   ]))

    # ── crises and years ──
    BN = A6["bench"]
    cols = [(fin["GR1"], GR["GR1"][0]), (fin["GR2"], GR["GR2"][0]), (fin["GR3"], GR["GR3"][0]), (S9, "Monthly")]
    rows = []
    for k, lab_ in CRISIS_LABEL.items():
        cells = [lt(lab_)]
        for n, _ in cols:
            v = K[n]["q"]["crises"][k]
            cells.append(td(f'{sg(v)} <span class="cap">({sg(K[n]["crises_dd"][k])})</span>', "neg" if v < 0 else "pos"))
        for b_ in ("S&P 500", "QQQ"):
            v = BN[b_]["crises"][k]
            cells.append(td(sg(v), "neg" if v < 0 else "pos"))
        rows.append(tr(cells))
    for i, cf in enumerate(K[fin["GR1"]]["cofalls"]):
        cells = [lt(f'Co-fall {cf["start"][5:7]}/{cf["start"][:4]}')]
        for n, _ in cols:
            v = K[n]["cofalls"][i]["book"]
            cells.append(td(sg(v), "neg" if v < 0 else "pos"))
        key = next(kk for kk in BN["S&P 500"]["crises"] if kk.startswith(f'cofall|{cf["start"]}'))
        for b_ in ("S&P 500", "QQQ"):
            v = BN[b_]["crises"][key]
            cells.append(td(sg(v), "neg" if v < 0 else "pos"))
        rows.append(tr(cells))
    out["CRISES"] = (table(["Event"] + [c for _, c in cols] + ["S&amp;P 500", "QQQ"], rows)
                     + f'<p class="cap">בכל תא: התשואה לאורך החלון, ובסוגריים הירידה הגרועה בתוכו. לפני {L("10/2012")} ה־{L("TAA")} הוא פרוקסי סינתטי, כולל כל 2008. ״{L("Co-fall")}״ = חלונות של 21 יום שבהם גם מניות וגם אג״ח ירדו.</p>')
    years = sorted(K[fin["GR1"]]["years"])
    rows = [tr([td(f'{y}{"*" if y in ("2008", "2026") else ""}')] + [td(sg(K[n]["years"][y]), "neg" if K[n]["years"][y] < 0 else "") for n, _ in cols]
               + [td(sg(BN[b_]["years"].get(y)), "neg" if (BN[b_]["years"].get(y) or 0) < 0 else "") for b_ in ("S&P 500", "QQQ")]) for y in years]
    out["YEARS"] = table(["Year"] + [c for _, c in cols] + ["S&amp;P 500", "QQQ"], rows) + '<p class="cap">* שנה חלקית.</p>'

    # ── alpha ──
    AL = B["alpha"]

    def a_cell(book: str, basis: str, window: str, model: str) -> str:
        r = next((a for a in AL if a["book"] == book and a["basis"] == basis and a["window"] == window and a["model"] == model), None)
        if r is None:
            return td("–")
        return td(f'{sg(r["alpha_ann"])} <span class="cap">(t {r["alpha_t"]:.1f})</span>', "pos" if r["alpha_t"] >= 2 else "neg" if r["alpha_t"] < 1 else "")
    rows = []
    for n, lab_ in ((fin["GR1"], GR["GR1"][0]), (fin["GR2"], GR["GR2"][0]), (fin["GR3"], GR["GR3"][0]), (S9, "Monthly"), ("capsule TAA 3x", "TAA 3x alone"),
                    ("capsule MOM", "Momentum capsule alone"), ("capsule MR", "MR capsule alone")):
        for basis in ("gross", "net"):
            rows.append(tr([lt(f"{lab_}, {basis}")] + [a_cell(n, basis, "long", m) for m in ("M0", "M1", "M2")] + [a_cell(n, basis, "h1", "M2"), a_cell(n, basis, "h2", "M2"),
                                                                                                                   a_cell(n, basis, "exact", "M2")], "dim" if basis == "net" else ""))
    a1 = next(a for a in AL if a["book"] == fin["GR1"] and a["basis"] == "net" and a["window"] == "long" and a["model"] == "M2")
    a1g = next(a for a in AL if a["book"] == fin["GR1"] and a["basis"] == "gross" and a["window"] == "long" and a["model"] == "M2")
    out["ALPHA_INTRO"] = points([
        f"<b>השיטה:</b> רגרסיה שבועית, {L('Newey-West')} עם 4 פיגורים.",
        f"<b>{L('Growth')}:</b> אחרי {L('QQQ')}, סל ה־{L('ETF')} וכלל ה־200 יום של {L('QQQ')} נשארת אלפא ברוטו של {L(f'{sg(a1g['alpha_ann'])} (t {a1g['alpha_t']:.1f})')}, ונטו אחרי {L('2/20')} של {L(f'{sg(a1['alpha_ann'])} (t {a1['alpha_t']:.1f})')}.",
        "<b>כל המספרים בתוך המדגם.</b>",
    ])
    out["ALPHA"] = table(["Book, basis", "LONG: vs QQQ", "LONG: vs ETF mix", "LONG: vs ETF mix + QQQ 200d rule", "H1 (to 2017-06): full model", "H2: full model", "2012+: full model"], rows)

    # ── capacity and ease ──
    CB = C.get("books", {})
    ez1, ezm = CB.get(fin["GR1"], {}).get("ease", {}), CB.get(S9, {}).get("ease", {})
    out["CAP_INTRO"] = points([
        "<b>כל המספרים לפני מדידת עלות אמיתית.</b>",
        f"<b>שני סרגלים:</b> מודל הבית (מחמיר מאוד בפתיחה, עד {L('0.05%')} מהמחזור) וסינון השתתפות לפי רגל (מתי פקודה גדולה שווה {L('5%')} מיום מסחר חציוני). שניהם אומרים דבר דומה על הפתיחה: כמה מיליונים.",
        f"<b>מה מגביל את מוצרי הקפסולות:</b> במודל הבית, המניות של ה־{L('MR')} ({L('FOX, NWS')}). בסינון ההשתתפות, {L('BTAL')} בתוך ה־{L('TAA')}, שצריך לעבוד אותו בכל גודל מעל כמה מיליונים. "
        f"נתיב העבודה שולח את מניות ה־{L('MR')} למכרז הסגירה. המנוע לא מדמה את זה, ולכן זה גבול עליון.",
        f"<b>התיק החודשי:</b> שני פודים, מסחר חודשי בלבד, ולכן אפשר לעבוד את הפקודות לאורך כמה ימים. {R}{L('Monthly')}: {L(capstr(S9, 'MOO'))} בפתיחה, {L(capstr(S9))} בנתיב עבודה. "
        f"חסר לו רק חיווט של {L(', '.join(h['POD'].get(a, a) for a in ezm.get('not_wired', [])) or 'none')}.",
    ])
    rows = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי"),
                  (S13, "החודשי מ־1 באוקטובר, ארבעה פודים (להשוואה)"), (GR1_L, f"{R}{L('Growth')} בלי {L('MR')}"),
                  ("leg TAA 3x", ""), ("leg TAA 3x 1N", ""), ("leg CORE5", ""), ("leg MOM", ""), ("leg MR", "")):
        e = CB.get(n)
        if not e:
            continue
        rt = e["routes"]
        cell = lambda r: td(f'{usd(rt[r]["recommended"], rt[r]["at_grid_top"]) if rt[r]["recommended"] else "<$0.5M"} <span class="cap">{_html.escape(rt[r]["first_fail"][:44])}</span>')  # noqa: E731
        pl = e.get("participation_by_leg")
        pa = e["participation"]
        p99 = f'{usd(pl["aum_p99_at_5pct"]["aum"])} <span class="cap">{pl["aum_p99_at_5pct"]["binding_leg"].replace("leg ", "")}</span>' if pl else usd(pa["aum_p99_at_5pct"])
        p90 = f'{usd(pl["aum_p90_at_5pct"]["aum"])} <span class="cap">{pl["aum_p90_at_5pct"]["binding_leg"].replace("leg ", "")}</span>' if pl else usd(pa["aum_p90_at_5pct"])
        rows.append(tr([name_cell(he or n), cell("MOO"), cell("MOC"), cell("worked+blocks"), td(p90), td(p99),
                        td(f'{usd(pa["aum_max_at_5pct"])} <span class="cap">{pa.get("binding_symbol", "")}</span>'), td(usd(e["btal_wall"]) if e["btal_wall"] else "–"),
                        td('<span class="no">CAPACITY-LIMITED</span>' if e["capacity_limited"] else '<span class="yes">ok</span>')]))
    out["CAPACITY"] = table(["Book", "House model: open (the only route the backtest models)", "House model: close auction", "House model: worked + blocks (upper bound)",
                             "Participation per leg: P90 order = 5% of a median day (binding leg)", "P99 order = 5% (binding leg)", "Largest single order, same-day orders of all pods summed = 5%", "BTAL 10%-ownership wall",
                             "AUM rule ($25M, worked route)"], rows)
    rows = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")):
        e = CB.get(n)
        if not e:
            continue
        ez, fi = e["ease"], e["fee_income"]
        rows.append(tr([name_cell(he), td(f'{ez["research_pods"]} / {ez["live_pods"]}'), td(str(ez["margin_accounts"])), td(f'{ez.get("trade_days_per_year", 0):.0f}'),
                        td(f'{e["orders_per_year"]:.0f}'), lt(", ".join(h["POD"].get(a, a) for a in ez["not_wired"]) or "none"), td(usd(ez["min_clean_size"])),
                        td(f'{pct(fi["total_pct"])} <span class="cap">({pct(fi["mgmt_pct"])} + {pct(fi["perf_pct"])})</span>')]))
    out["EASE"] = table(["Book", "Pods: in research / in the live build", "Margin accounts", "Trade days / yr", "Orders / yr", "Not wired at 5c0d48d", "Minimum clean size",
                         "2/20 income, % of AUM a year (mgmt + perf)"], rows)

    # ── important to know ──
    inp = S["inputs"]
    cash_gap = lambda n: K[n]["q"]["cagr"] - K[n]["frames"]["s1_house_cash"]["cagr"]  # noqa: E731
    mr5 = K["capsule MR"]["frames"]
    know = [
        ("המנוע משלם 0% על מזומן פנוי; במספרי הכותרת מזומן מרוויח ריבית", "con",
         f"{R}{L('Growth')}: {L(sg(cash_gap('GR1'), 2))} לשנה. {R}{L('TAA 3x')} מחזיק רק כ־{L('2%')} מזומן ({L(sg(cash_gap('capsule TAA 3x'), 2))}), כך שב־{L('TAA')} זה זניח; במומנטום (רבע עד שליש פנוי) {L(sg(cash_gap('capsule MOM'), 2))}; ב־{L('MR')} (מזומן שיורי ליד ה־{L('BIL')}) {L(sg(cash_gap('capsule MR'), 2))}.",
         "g_lib.frames; inputs audit"),
        ("המוצרים לא מושקעים עד הסוף", "neu",
         f"בממוצע {L(pct(XB['GR1']['tbill_like_share']['mean'], 0))} מ־{L('Growth')} ב־{L('BIL')} או במזומן ({L(pct(XB['GR1']['tbill_like_share']['gate_closed'], 0))} כשהשער סגור). לקוח משלם עמלה גם על החלק הזה.",
         "exposure.py"),
        (f"שלוש דרכים שונות לחשב מזומן: ה־{L('BIL')} שה־{L('MR')} מחזיק הוא פוזיציה אמיתית (ניכוי במקור {L('25%')} ועלות מסחר); מזומן פנוי ב־{L('TAA')} ובמומנטום מקבל {L('DTB3 − 0.5%')} בלי עלות; רגל המזומן היא {L('BIL total return')}", "con",
         f"כ־{L('0.6 pp')} מהתשואה של הקפסולה; ב־{L('Growth')}: {L(sg(K['GR1']['frames']['s9_mr_fair_cash']['cagr'] - K['GR1']['q']['cagr'], 2))} אם ה־{L('MR')} היה מקבל את אותה ריבית כמו האחרים.",
         "frame s9_mr_fair_cash"),
        (f"לפני {L('10/2012')} ה־{L('TAA')} הוא פרוקסי סינתטי של {L('TQQQ')} ו־{L('BTAL')}", "unk",
         f"רבע מהמדגם, וה־2008 היחיד. בלי קנה המידה השמרני: {L(f'CAGR {pct(K['GR1']['frames']['s2_proxy_unscaled']['cagr'])}')}. מ־2012 בלבד: {L(f'{pct(K['GR1']['frames']['s6_exact']['cagr'])} / {f2(K['GR1']['frames']['s6_exact']['xs'])}')}.",
         "frames s2, s6"),
        (f"עמלות ה־{L('TAA')} מחושבות על מניות {L('TQQQ')} מותאמות פיצול", "con", f"כ־{L('0.3 pp')} לשנה לפוד בממוצע 2012 עד 2026 ({L('0.38')} בגרסת {L('1N')}), כמעט אפס היום.", "docs/strategies/taa-defensive-tqqq.md"),
        ("מזומן שלילי לא ממומן במנוע", "opt",
         f"קטן. פודי ה־{L('MR')}: {L(str(inp['dv2_g']['negative_cash_day_count_int']))} ו־{L(str(inp['hpi_g']['negative_cash_day_count_int']))} ימים, עד {L(sg(inp['dv2_g']['minimum_cash_nav_weight_float']))} מהפוד. "
         f"פודי המומנטום: כ־{L(str(inp['ndx_atr_cap']['negative_cash_day_count_int']))} ימים, עד {L(sg(inp['ndx_atr_cap']['minimum_cash_nav_weight_float']))}; ב־{L('TAA')} כ־{L('1,500')} ימים (מ־2012), עד {L('−1.8%')}. במספרי הכותרת זה מחויב ב־{L('DTB3 + 1.5%')}.",
         "sources metadata"),
        (f"חשיפה לנאסד״ק דרך {L('TQQQ')}; שניים משלושת המנועים לונג נאסד״ק כשהשוק חיובי; אין במדגם דוב כמו 2000 עד 2002", "opt",
         f"שיא {L(f'{XB['GR1']['nasdaq_lookthrough']['max']:.2f}×')} ב־{L('Growth')}, {L(f'{XB['GR3']['nasdaq_lookthrough']['max']:.2f}×')} ב־{L('Aggressive')} במשקלי היעד "
         f"({L(f'{XB['GR1']['nasdaq_lookthrough_drift']['max']:.2f}×')} ו־{L(f'{XB['GR3']['nasdaq_lookthrough_drift']['max']:.2f}×')} עם הסחיפה). יום של {L('−10%')} בנאסד״ק בשיא החשיפה: {L(sg(XB['GR1']['gap_table']['10%']['peak']))} ו־{L(sg(XB['GR3']['gap_table']['10%']['peak']))}.",
         "exposure.py"),
        ("כל המנועים נבחרו על אותה היסטוריה; אין מספר מחוץ למדגם", "opt",
         f"{R}{L('MR')}: כ־{L('110')} גרסאות נבדקו; מומנטום: כ־{L('45')} ניסיונות ועוד רשת של {L('151')} תצורות. בתרחיש השמרני (שלושה רבעים מהיתרון) {L('Growth')} הוא {L(f'CAGR {pct(cg1.get('cagr'))} / Excess Sharpe {f2(cg1.get('xs'))}')} "
         f"מול {L(f'{pct(g1['q']['cagr'])} / {f2(g1['q']['xs'])}')} ב־{L('backtest')}. ה־{L('Excess Sharpe')} שלו היה {L(f2(xb_))} ב־2012 עד 2021 ו־{L(f2(xc_))} מאז 2022.",
         "SPEC section 0; battery.py"),
        ("משקלי המוצרים נבחרו על ידך אחרי התוצאות, מתוך נקודות חוגת ה־TAA שנרשמו מראש", "opt",
         f"{R}{L('Growth')} ו־{L('Growth Plus')} הם {L('40 / 30 / 30')} ו־{L('Aggressive')} הוא {L('60 / 20 / 20')}; ברירת המחדל הרשומה הייתה הון שווה. מול ההון השווה: {tie0} ב־{L('Sharpe')} ({L(pct(v0['share_xs'], 0))} מהמסלולים). נקודת {L('40 / 30 / 30')} לא הייתה ברשת שנראתה לפני ההקפאה; היא נבחרה אחרי התוצאות המלאות, מתוך מפת החוגה שנרשמה מראש.",
         "SPEC amendments O1, O2"),
        ("התיק החודשי תוכנן אחרי התוצאות, מרשת גישוש שלא נרשמה מראש", "opt",
         f"שני סוגי {L('TAA')}, ארבעה יחסים מול {L('CORE5')}, מומנטום {L('0 / 15 / 30%')}; ההבדלים בין שורות שכנות קטנים. את היחס {L(m_short)} בחרת אחרי התוצאות. "
         f"מנוע תשואה אחד: עם {L('TAA')} מת נשאר {L(f'Excess Sharpe {f2(d_s9)}')}.",
         "monthly.py; SPEC amendment O3"),
        (f"התיק החודשי על מדרגת {rung_m or 'none'}, ומרווח המדרגה שלו", "opt",
         f"{R}{monthly_margin} (ב־{L('Growth')}: {L(f2(em_k('GR1', 'p20')))})." + (f" זו מדרגה גבוהה מזו של {L('Growth')}: יותר סיכון, לא פחות." if more_risk_m else ""),
         "battery.py edge margin"),
        (f"המדרגה {L('AGGRESSIVE')} חדשה במחקר הזה", "neu", agg_ok_txt + f" {R}{L('Aggressive')} עובר רק אותה.", "SPEC; owner decision"),
        ("המינוף הוא חלופה, לא מוצר", "opt", f"{R}{L('Growth × 1.40')}: מימון {L('DTB3 + 1.5%')}, בלי קריאות מרג׳ין ובלי יום פער; לא זמין כל עוד כל פוד בחשבון {L('Reg-T')} נפרד.", "study.py margin"),
        (f"מספר החריגה {L('DD beyond −X%')} הוא מוסכמה (בלוק 63, יתרון מלא)", "opt",
         f"{R}{L('Growth')}, {L('−20%')}: {L(pct(g1['tails']['p20']))} לפי הכלל, {L(pct(BT['blocks']['21'][fin['GR1']]['p20']))} בבלוק 21, {L(pct(ED['scenarios']['all at 0.75'][fin['GR1']]['p20']))} בשלושה רבעים מהיתרון.",
         "battery.py"),
        (f"ה־{L('MR')} רגישה לעלות ואין לה עסקאות אמיתיות", "opt",
         f"הקפסולה לבדה: {L(pct(K['capsule MR']['q']['cagr']))} ← {L(pct(mr5['s3_plus_5bps']['cagr']))} ב־{L('+5 bps')} ← {L(pct(mr5['plus_10bps']['cagr']))} ב־{L('+10 bps')}. ב־{L('Growth')}: {L(pct(g1['q']['cagr']))} ← {L(pct(g1['frames']['s3_plus_5bps']['cagr']))} ← {L(pct(g1['frames']['plus_10bps']['cagr']))}. מחזור מניות של כ־{L(f"{CB['leg MR']['stock_turnover_x_nav']:.0f}×")} הפוד בשנה (שלוש השנים האחרונות). גם: 2020 ו־2021 הם כשליש מהרווח שלה. שער העלות ({L('4 bps')} על 200 עסקאות) עדיין לא נמדד.",
         "frames s3; MR audit"),
        ("קפסולת המומנטום", "unk",
         f"תוויות {L('GICS')} של היום על כל ההיסטוריה (מעט אופטימי). הבחירה במניות לא הוכחה מול {L('QQQ')} באותה חשיפה. היא בירידה של כ־{L('18%')} מהשיא ביוני 2026 (נכון ל־2 באוקטובר; {L('−16.5%')} בסוף החלון). במבחן המקום: עם {L('BIL')} במקומה ה־{L('Sharpe')} של התיק גבוה יותר ב־{L(pct(1 - t1['main']['share_xs'], 0))} מהמסלולים.",
         "MOMENTUM_DECISION_20261004.md"),
        ("האיפוס השנתי הוא העברה בלי עלות, בתאריך אחד", "opt",
         f"קטן: טווח ה־{L('CAGR')} על 12 חודשי התחלה ב־{L('Growth')} הוא {L(f'{pct(RS['GR1']['start_month_spread']['cagr']['min'])} … {pct(RS['GR1']['start_month_spread']['cagr']['max'])}')}.",
         "battery.py reset"),
        ("גודל: לפני מדידה, ותלוי בנתיב הביצוע", "unk",
         f"{R}{L('Growth')}: {L(capstr(fin['GR1'], 'MOO'))} בפתיחה ו־{L(capstr(fin['GR1']))} בנתיב עבודה (מודל הבית; מגבילות מניות ה־{L('MR')}: {L('FOX, NWS')}); בסינון השתתפות לפי רגל {L(usd(CB[fin['GR1']]['participation_by_leg']['aum_p99_at_5pct']['aum']))} "
         f"(מגביל {L('BTAL')}, שצריך לעבוד אותו). הנתיב של ה־{L('MR')} בסגירה לא נבדק במנוע, ומחקר התזמון של {L('DV2')} דחה סגירה באותו יום.",
         "capacity.py"),
        ("גודל מינימלי", "neu",
         f"{R}{L('Growth')} צריך כ־{L(usd(ez1.get('min_clean_size')))} כדי שכל פוד יהיה נקי, והתיק החודשי כ־{L(usd(ezm.get('min_clean_size')))}." + (" שניהם גדולים מהחשבון של היום." if min(ez1.get('min_clean_size') or 0, ezm.get('min_clean_size') or 0) > 30_000 else ""), "capacity.py ease"),
        ("מה מחווט", "neu",
         f"מחווטים: {L('TAA 3x, TAA 3x 1N, BTAL_QQQ, NDX-VXN')}; מהם רק {L('TAA 3x')} רץ היום בפועל (בחשבון שלך), ו־{L('TAA 3x 1N')} מחווט אבל לא רץ. לא מחווטים: {L('CORE5')} ({L('PM_READY')}) וארבעת הפודים של המומנטום וה־{L('MR')} ({L('PM_READY')} בלי נתיב לייב, נכון ל־{L('5c0d48d')}); ל־{L('MR')} צריך צורת פקודה חדשה ושני חשבונות מרג׳ין. "
         f"לתיק החודשי חסר רק: {L(', '.join(h['POD'].get(a, a) for a in ezm.get('not_wired', [])) or 'none')}.",
         "strategy_registry.py"),
        ("שורות המינוף", "opt", f"מימון {L('DTB3 + 1.5%')}; אין קריאות מרג׳ין ואין יום פער במודל.", "study.py margin"),
        (f"ברוטו מול נטו {L('2/20')}", "neu",
         f"{R}{L('Growth')}: {L(pct(g1['q']['cagr']))} ברוטו, {L(pct(g1['net']['cagr']))} נטו. בלי הוצאות קרן (אדמין, ביקורת).", "ga_lib.fee_nav"),
        (f"בסיס ה־{L('Sharpe')}", "neu",
         f"{R}{L('Excess Sharpe')} מעל {L('BIL')} על תשואות יומיות. {L('Growth')}: {L(f2(g1['q']['xs']))} יומי, {L(f2(g1['q']['xs_monthly']))} חודשי, {L(f2(g1['q']['sharpe0']))} בריבית אפס (כמו ב־{L('Bench')}).",
         "g_lib.stats"),
        ("השבועות אחרי סוף החלון אינם מבחן קדימה", "neu",
         f"בין {L('20.8.2026')} ל־{L('2.10.2026')}: {L('Growth')} {L(sg(B['after_window']['books'][fin['GR1']]['after_window']))}, החודשי {L(sg(B['after_window']['books'][S9]['after_window']))}. שתי הקפסולות תוכננו על נתונים עד 2 באוקטובר.",
         "battery.py after_window"),
        (f"פוד ה־{L('BIL')} במנוע מול {L('BIL total return')}", "neu", f"ב־{L('Bench')} פוד {L('BIL')} מרוויח פחות (ניכוי במקור, בלי השקעה מחדש): {L('1.08%')} מול {L('1.61%')} לשנה מ־2012.", "rerun audit"),
    ]
    dir_lab = {"con": "conservative", "opt": "optimistic", "unk": "unknown", "neu": "note"}
    rows = [tr([f'<td class="he">{a}</td>', f'<td><span class="dir {d}">{dir_lab[d]}</span></td>', f'<td class="he">{b}</td>', f'<td class="cap">{c}</td>']) for a, d, b, c in know]
    out["KNOW"] = table(["Convention or caveat", "Direction", "Measured size", "Source"], rows)

    # ── what is needed ──
    code_ = lambda s_: f'<span class="code">{s_}</span>'  # noqa: E731
    out["TODO"] = "".join(f"<li>{rl(x)}</li>" for x in [
        f"<b>הוחלט (5 באוקטובר), מוצרי הקפסולות:</b> {L('Growth')} = {L('TAA 3x 40 / 30 / 30')} ({code_('fund_growth.yaml')}), המוצר שבונים אליו. {R}{L('Growth Plus')} = אותם משקלים עם {L('TAA 3x 1N')} ({code_('fund_growth_plus.yaml')}).",
        f"<b>הוחלט (5 באוקטובר):</b> {L('Aggressive')} = {L('TAA 3x 1N 60 / 20 / 20')} ({code_('fund_growth_aggressive.yaml')}). הסולם: {L('TAA 3x 40')} ← {L('TAA 3x 1N 40')} ← {L('TAA 3x 1N 60')}.",
        f"<b>הוחלט (5 באוקטובר), התיק החודשי:</b> {L('Monthly')} = {L(m_mix)} ({code_('fund_growth_monthly.yaml')}). שני פודים; רץ ראשון, והוא החלופה לגודל. {R}{L('Monthly Plus')} בוטל.",
        f"<b>הוחלט (5 באוקטובר):</b> {agg_ok_txt}",
        f"<b>הוחלט:</b> מספר הכותרת הוא ה־{L('backtest')}; תרחיש שמרני אחד ותרחיש לחץ אחד בפרק ״כמה להאמין״.",
        f"<b>חלופה, לא החלטה:</b> {L('Growth × 1.40')} כדרך אחרת לרמת התשואה של {L('Aggressive')}; רלוונטי רק בקרן עם חשבון מרג׳ין אחד.",
        f"<b>פתוח: שער העלות של ה־{L('MR')}:</b> להריץ את הקפסולה בנייר או בקטן ולמדוד: עד {L('4 bps')} לצד על 200 עסקאות. זה הניסוי היחיד שנותן ראיה נקייה. "
        f"אם השער נכשל, מוצר הצמיחה נשאר התיק החודשי וצריך תוכנית חדשה.",
        f"<b>פתוח: סדר החיווט:</b> ראשון {L('CORE5')} (נחוץ להשקה ההגנתית ולתיק החודשי; אחריו {L('Monthly')} יכול לרוץ). שני ה־{L('MR')} (צורת פקודה חדשה, בעבודה), קודם בנייר. שלישי ולא דחוף: אסטרטגיית מומנטום אחת שממצעת את שני הספרים. "
        f"עם כלל ה־{L('NDX')} החי במקומה {L('Growth')} נותן כמעט אותה תוצאה (מתאם {L(f2(S['standins']['GR1 with live NDX rule']['corr_with_gr1']))}).",
        f"<b>פתוח: נתיב הביצוע של ה־{L('MR')}:</b> סגירה או פתיחה. במודל הבית זה ההבדל בין {L(capstr(fin['GR1'], 'MOO'))} ל־{L(capstr(fin['GR1']))} ל־{L('Growth')}. מעבר לזה, התיק החודשי.",
        f"<b>פתוח: {L('commit')} ומיזוג:</b> המחקר יושב בענף עבודה ולא מוזג לענף הראשי.",
    ])

    out["TRIGGERS"] = '<p class="cap"><b>טריגרים לבדיקה מחדש</b> (פותחים דיון, לא יציאה אוטומטית):</p>' + notes([
        f'ירידה של קפסולה מעבר למקסימום שנקבע לה בתוכנית ({L("MOM −30%, MR −21%, TAA −26%")}).',
        f'עלות ביצוע של ה־{L("MR")} מעל {L("4 bps")} לצד על 200 עסקאות.',
        f'קפסולה שמפגרת אחרי {L("BIL")} שלוש שנים מתגלגלות.',
    ])

    rows_c = []
    for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")):
        if n not in CB:
            continue
        lv = CB[n]["routes"]
        rows_c.append(tr([name_cell(he)] + [td(" / ".join(pct(lv[r_]["levels"][a_]["cost"], 2) + ("" if lv[r_]["levels"][a_]["gates_ok"] else " ✗") for a_ in ("10M", "25M", "50M")))
                                            for r_ in ("MOO", "MOC", "worked+blocks")] + [td(pct(CB[n]["exact_excess_cagr"]))]))
    out["CAPACITY"] += ("<details><summary>עלות שנתית לפי גודל ונתיב</summary>"
                        + table(["Book: cost % of NAV a year at $10M / $25M / $50M (✗ = a gate fails)", "Open", "Close auction", "Worked + blocks",
                                 "2012+ excess CAGR (the cost cap is 25% of it)"], rows_c) + "</details>")

    # ── verification ──
    rows = []
    for code, he_ in [(c_, GR[c_][1]) for c_ in GR] + [("M1", MONTHLY_HE[S9])]:
        p = (P.get("books") or {}).get(code)
        if not p or "error" in p or "engine" not in p:
            rows.append(tr([name_cell(he_), td("pending"), td("–"), td("–"), td("–"), td("–")]))
            continue
        rows.append(tr([name_cell(he_), td(f'{pct(p["engine"]["cagr"], 2)} / {sg(p["engine"]["dd"])}'), td(f'{pct(p["research_house_cash"]["cagr"], 2)} / {sg(p["research_house_cash"]["dd"])}'),
                        td(f'{p["corr"]:.4f}'), td(f'{p["max_abs_daily_diff"] * 100:.2f}%'), td(yes(p["accepted"]))]))
    aw = B["after_window"]
    rows2 = [tr([name_cell(he), td(sg(aw["books"][n]["after_window"])), td(sg(aw["books"][n]["after_window_dd"])), td(sg(aw["books"][n]["ytd_2026"]))])
             for n, he in ((fin["GR1"], GR["GR1"][1]), (fin["GR2"], GR["GR2"][1]), (fin["GR3"], GR["GR3"][1]), (S9, "החודשי")) if n in aw["books"]]
    slv = aw["sleeves"]
    out["VERIFY"] = (
        "<ul class=\"plain\">"
        f"<li><b>סדרות קיימות:</b> {L('TAA 3x, TAA 3x 1N, CORE5, BTAL_QQQ, NDX-VXN, DV2-IND, EOM, downshock')} הורצו מחדש על הקוד הנוכחי ויצאו זהות בית לבית לקבצים השמורים. "
        f"{R}{L('DV2')} ו־{L('HPI')} שונים בעד {L('0.0055%')} ליום (עדכון נתונים של {L('Norgate')}), בתוך הסף.</li>"
        f"<li><b>הקפסולות:</b> ספר {L('E2')} זהה ביט לביט להרצה מ־4 באוקטובר. קפסולת ה־{L('MR')} ב־{L('$1M')} לפוד: {L('CAGR')} בטווח {L('0.05 pp')} מההרצה השמורה, מתאם {L('0.9996')}.</li>"
        f"<li><b>סיבתיות:</b> הרצה שמסתיימת מאוחר יותר משחזרת בדיוק את התאריכים המוקדמים (שני מבחנים).</li>"
        f"<li><b>המחקר הקודם:</b> התיק החודשי מ־1 באוקטובר (ארבעה פודים) שוחזר בדיוק ({L(f'DD beyond −20% {pct(S['checks']['incumbent_p20'][1], 2)}')} בשתי ההרצות).</li>"
        f"<li><b>שחזור הקבצים ההגנתיים:</b> {L('a6d.py')}: {L('1,011')} שדות, {L('0')} הבדלים; "
        f"{L('report_a6.py')}: {L('14,899')} שדות, {L('0')} הבדלים. {R}{L('a6.py')}: 7 שדות של ירידה בתוך חלון המכסים של 2025 בשורות הצמיחה הישנות שונים בעד {L('0.36 pp')}, "
        f"כי {L('a6.json')} נכתב לפני תיקון קטן בקוד באותו יום; אף שדה הגנתי לא שונה.</li>"
        f"<li><b>העץ הראשי:</b> יש בו שינויים לא מחויבים של סשן אחר (חיווט); הם לא משפיעים על בקטסט, ואף סדרה לא יוצרה ממנו. כל ההרצות כאן מעץ נקי.</li></ul>"
        + f"<h3 class=\"sub\">אישור מנוע: קובצי ה־{L('YAML')} דרך ה־{L('PortfolioManager')} מול מודל המחקר (מזומן 0%, מ־2013)</h3>"
        + table(["Product", "Engine: CAGR / Max DD", "Research book model", "Daily correlation", "Largest daily difference", "Accepted (0.3 pp / 1 pp / 0.995)"], rows)
        + f"<h3 class=\"sub\">אחרי סוף החלון ({L(aw['window'][0] + ' → ' + aw['window'][1])}; לא מבחן קדימה)</h3>"
        + table(["Book", "Return after the window", "Worst drawdown inside", "2026 to date"], rows2)
        + f'<p class="cap">{R}{L("2026 to date")} מחושב בנפרד, מסדרות ההרצה המאוחרת (עד 2 באוקטובר). הוא שונה במעט, עד {L("0.3 pp")}, משרשור של טבלת השנים; לא ביררתי את מקור ההפרש.</p>'
        + f'<p class="cap">לפי רגל: {L(f"TAA 3x {sg(slv['taa3x'])}, TAA 3x 1N {sg(slv['taa3x_1n'])}, NDX ATR cap {sg(slv['ndx_atr_cap'])}, NDX NATR cap {sg(slv['ndx_natr_cap'])}, DV2-G {sg(slv['dv2_g'])}, HPI-G {sg(slv['hpi_g'])}")}.</p>')

    # ── hedge fund: fee income per schedule (fees.py; owner request 2026-10-05) ──
    FE = h.get("FEES") or {}
    if FE.get("books"):
        FB = FE["books"]
        sch = [f"{m * 100:g}/{p_ * 100:g}" for m, p_ in FE["schedules"]]
        lab_s = {k: k.replace("/", " / ") for k in sch}
        prod = [n for n in ("Defensive launch", "Blend Growth 40 / Defensive 60", "Growth", "Growth Plus", "Aggressive", "Growth x1.40 (leverage)", "Monthly") if n in FB]
        he_p = {"Defensive launch": "ההשקה ההגנתית", "Blend Growth 40 / Defensive 60": f"{R}{L('Growth 40 / Defensive 60')}", "Growth": "צמיחה", "Growth Plus": "צמיחה פלוס", "Aggressive": "אגרסיבי",
                "Monthly": "החודשי", "Growth x1.40 (leverage)": f"{R}{L('Growth × 1.40')} (מינוף, חלופה)"}
        k_ = lambda x: f"${x * 1e6 / 1e3:.0f}K"  # noqa: E731   income on $1M of AUM
        rows = [tr([name_cell(he_p[n], n)] + [td(f'{k_(FB[n]["backtest"][c]["income_mean"])} <span class="cap">{k_(FB[n]["conservative"][c]["income_mean"])}</span>') for c in sch]) for n in prod]
        t_inc = table(["Average income a year on $1M: backtest, then conservative case"] + [lab_s[c] for c in sch], rows)
        rows = [tr([name_cell(he_p[n], n), td(f'{pct(FB[n]["backtest"][sch[0]]["gross_cagr"])} / {f2(FB[n]["backtest"][sch[0]]["gross_xs"])}')]
                   + [td(f'{pct(FB[n]["backtest"][c]["net_cagr"])} / {f2(FB[n]["backtest"][c]["net_xs"])}') for c in sch]) for n in prod]
        t_net = table(["What the client keeps: CAGR / Excess Sharpe (backtest)", "Before fees"] + [lab_s[c] for c in sch], rows)
        rows = [tr([name_cell(he_p[n], n)] + [td(f'{pct(FB[n]["backtest"][c]["share_of_excess_over_bil"], 0)} <span class="cap">{pct(FB[n]["conservative"][c]["share_of_excess_over_bil"], 0)}</span>') for c in sch]) for n in prod]
        t_share = table(["Share of the return above T-bills that the fee takes: backtest, then conservative"] + [lab_s[c] for c in sch], rows)
        fg = FB["Growth"]["backtest"]
        rows = [tr([lt(lab_s[c]), td(k_(fg[c]["mgmt_mean"])), td(k_(fg[c]["perf_mean"])), td(k_(fg[c]["income_mean"])), td(k_(fg[c]["income_median"])),
                    td(f'{k_(fg[c]["income_min"])} ({fg[c]["worst_year"]})'), td(k_(fg[c]["income_max"])), td(f'{fg[c]["years_no_perf_fee"]} of {fg[c]["years"]}'),
                    td(f'${100e3 / fg[c]["income_mean"] / 1e6:.1f}M'), td(f'${100e3 / FB["Growth"]["conservative"][c]["income_mean"] / 1e6:.1f}M')]) for c in sch]
        t_g = table(["Growth on $1M, by fee schedule", "Management fee", "Performance fee (average)", "Total, average year", "Median year", "Worst year", "Best year",
                     "Years with no performance fee", "AUM for $100K a year: backtest", "conservative"], rows)
        gx = lambda n, c, case="backtest": FB[n][case][c]  # noqa: E731
        spx, qqq = S["bench"]["SPXTR"]["xs"], S["bench"]["QQQ"]["xs"]
        out["FEES"] = (
            notes([
                f"<b>ההנחה:</b> {L('$1M')} מנוהל, קבוע. ההכנסה ליניארית בגודל: ב־{L('$5M')} מכפילים בחמש.",
                f"<b>מודל העמלה:</b> דמי ניהול נצברים יומית; דמי הצלחה על רווח מעל השיא הקודם ({L('high-water mark')}), פעם בשנה, בלי רף תשואה.",
                f"<b>שני מספרים בכל תא:</b> לפי ה־{L('backtest')}, ולידו באפור התרחיש השמרני (שלושה רבעים מהיתרון).",
                "<b>בלי הוצאות קרן:</b> אדמיניסטרציה, ביקורת ועורך דין לא נכללים. הם קבועים, ובגודל כזה הם יכולים לאכול את רוב ההכנסה.",
            ])
            + t_inc + t_net + t_share
            + f"<h3 class=\"sub\">{R}{L('Growth')}: ממה ההכנסה מורכבת, וכמה היא משתנה משנה לשנה</h3>" + t_g
            + "<h3 class=\"sub\">מה כדאי לגבות</h3>"
            + notes([
                f"<b>ההמלצה: {L('1 / 15')} למוצרי הצמיחה, {L('1 / 10')} להגנתי ולשילובים.</b> עם שיא קודם, בלי רף.",
                f"<b>הכלל שמאחוריה:</b> העמלה לוקחת בערך רבע מהתשואה מעל {L('T-bills')} לכל היותר, ובערך שליש בתרחיש השמרני. "
                f"ב־{L('Growth')} עם {L('1 / 15')}: {L(pct(gx('Growth', '1/15')['share_of_excess_over_bil'], 0))} ו־{L(pct(gx('Growth', '1/15', 'conservative')['share_of_excess_over_bil'], 0))}.",
                f"<b>למה לא {L('2 / 20')} ב־{L('Growth')}:</b> ללקוח נשאר {L(f'Excess Sharpe {f2(gx('Growth', '2/20')['net_xs'])}')}, ובתרחיש השמרני {L(f2(gx('Growth', '2/20', 'conservative')['net_xs']))}. "
                f"זה כבר קרוב למדד: {L(f'S&amp;P 500 {f2(spx)}, QQQ {f2(qqq)}')}. קשה למכור את זה בלי רקורד חי.",
                f"<b>כמה זה עולה לך:</b> ב־{L('$1M')} ההבדל בין {L('2 / 20')} ל־{L('1 / 15')} ב־{L('Growth')} הוא {L(k_(gx('Growth', '2/20')['income_mean'] - gx('Growth', '1/15')['income_mean']))} לשנה. "
                f"סכום קטן מול הסיכוי לגייס ולהחזיק לקוח.",
                f"<b>ההגנתי לא סובל עמלה גבוהה:</b> ב־{L('2 / 20')} העמלה לוקחת {L(pct(gx('Defensive launch', '2/20')['share_of_excess_over_bil'], 0))} מהתשואה מעל {L('T-bills')}, "
                f"וללקוח נשאר {L(f'CAGR {pct(gx('Defensive launch', '2/20')['net_cagr'])}')}. עם {L('1 / 10')}: {L(pct(gx('Defensive launch', '1/10')['share_of_excess_over_bil'], 0))} ו־{L(pct(gx('Defensive launch', '1/10')['net_cagr']))}.",
                f"<b>המנוף הוא הגודל, לא אחוז העמלה:</b> כדי להגיע ל־{L('$100K')} לשנה מ־{L('Growth')} צריך {L(f'${100e3 / gx('Growth', '2/20')['income_mean'] / 1e6:.1f}M')} ב־{L('2 / 20')} "
                f"ו־{L(f'${100e3 / gx('Growth', '1/15')['income_mean'] / 1e6:.1f}M')} ב־{L('1 / 15')}. מוצרי הקפסולות מוגבלים לכ־{L(capstr('GR1', 'MOO'))} בפתיחה; התיק החודשי גדל הרבה מעבר לזה.",
                f"<b>ההכנסה לא יציבה:</b> דמי הניהול קבועים, דמי ההצלחה לא. ב־{L('Growth')} עם {L('1 / 15')} שנה חלשה נותנת {L(k_(gx('Growth', '1/15')['income_min']))} ושנה חזקה {L(k_(gx('Growth', '1/15')['income_max']))}.",
                f"<b>{L('2 / 20')} כתקרה:</b> הגיוני רק אחרי רקורד חי של שנתיים־שלוש, או למוצר עם קיבולת מוגבלת שיש עליו ביקוש.",
                "<b>לפני הכול, עורך דין:</b> מי רשאי לגבות דמי הצלחה, וממי, תלוי ברישוי ובסוג המשקיע. המספרים כאן הם חשבון, לא מבנה משפטי.",
            ]))
    else:
        out["FEES"] = "<p>–</p>"

    out["REVIEW"] = REVIEW_HTML
    out["CAVEAT"] = notes([
        f"<b>סימולציה, 2008 עד 2026, לא מבחן עיוור.</b> מדרגות הסיכון והבדיקות נכתבו מראש. כל המנועים נבחרו על אותה היסטוריה. מספרי הכותרת הם ה־{L('backtest')}.",
        f"<b>משקלי המוצרים:</b> {L('Growth')} ו־{L('Growth Plus')} ({L('40 / 30 / 30')}) נבחרו על ידך אחרי התוצאות, מתוך נקודות חוגת ה־{L('TAA')} שהוגדרו מראש. גם {L('Aggressive')} ({L('60 / 20 / 20')}) נבחר על ידך אחרי התוצאות, מאותה חוגה.",
        f"<b>התיק החודשי</b> ({L(m_short)}) תוכנן אחרי התוצאות, מרשת גישוש שלא נרשמה מראש.",
        "<b>הראיה הנקייה היחידה שנשארה</b> היא מסחר בנייר או בלייב.",
    ])

    # ── charts ──
    def aligned(dates: list[str], pts: list) -> list:
        m = dict(pts)
        return [m.get(d) for d in dates]
    nav_dates = [d for d, _ in K[fin["GR1"]]["nav"]]
    charts = {"nav": {"dates": nav_dates, "series": [
        {"l": "Growth", "c": "var(--s3)", "pts": aligned(nav_dates, K[fin["GR1"]]["nav"])}, {"l": "Growth Plus", "c": "var(--s5)", "pts": aligned(nav_dates, K[fin["GR2"]]["nav"])},
        {"l": "Aggressive", "c": "var(--s4)", "pts": aligned(nav_dates, K[fin["GR3"]]["nav"])}, {"l": "Monthly", "c": "var(--s1)", "pts": aligned(nav_dates, K[S9]["nav"])},
        {"l": "S&P 500 TR", "c": "var(--bench)", "dash": True, "pts": aligned(nav_dates, S["bench_nav"]["SPXTR"])},
        {"l": "QQQ TR", "c": "var(--ink-3)", "dash": True, "pts": aligned(nav_dates, S["bench_nav"]["QQQ"])}]}}
    rs = B["rolling_series"]
    rd = [d for d, _ in rs[fin["GR1"]]]
    charts["roll"] = {"dates": rd, "series": [{"l": "Growth", "c": "var(--s3)", "pts": aligned(rd, rs[fin["GR1"]])}, {"l": "Monthly", "c": "var(--s1)", "pts": aligned(rd, rs[S9])},
                                              {"l": "TAA 3x alone", "c": "var(--bench)", "dash": True, "pts": aligned(rd, rs["capsule TAA 3x"])},
                                              {"l": "MOM alone", "c": "var(--s2)", "dash": True, "pts": aligned(rd, rs["capsule MOM"])},
                                              {"l": "MR alone", "c": "var(--s4)", "dash": True, "pts": aligned(rd, rs["capsule MR"])}]}
    rc = DP["rolling_252"]
    cd = [d for d, _ in rc["TAA 3x | MOM"]["series"]]
    charts["corr"] = {"dates": cd, "series": [{"l": "TAA 3x and MOM", "c": "var(--s3)", "pts": aligned(cd, rc["TAA 3x | MOM"]["series"])},
                                              {"l": "TAA 3x and MR", "c": "var(--s1)", "pts": aligned(cd, rc["TAA 3x | MR"]["series"])},
                                              {"l": "MOM and MR", "c": "var(--s4)", "pts": aligned(cd, rc["MOM | MR"]["series"])}]}
    nd = [d for d, _ in XB["GR1"]["nasdaq_monthly"]]
    charts["nasdaq"] = {"dates": nd, "series": [{"l": GR[c][0], "c": GR[c][2], "pts": aligned(nd, XB[c]["nasdaq_monthly"])} for c in GR]}
    for k_ in ("EYEBROW", "LEDE", "DEF_INTRO"):      # removed from the page at the owner's request (2026-10-05)
        out.pop(k_, None)
    return {"html": out, "charts": charts}


REVIEW_HTML = (
    "<ul class=\"plain\">"
    f"<li><b>הגרסה הזאת ({L('v5.2')}):</b> אחרי ההחלטות שלך מ־5 באוקטובר ({L('Growth 40 / 30 / 30')}, {L('Growth Plus 40 / 30 / 30')} עם {L('TAA 3x 1N')}, תיקים חודשיים של שני פודים, ה־{L('backtest')} כמספר הכותרת) "
    f"המחקר הורץ מחדש, והעמוד והרשומה נבדקו על ידי שני סוקרים בלתי תלויים. כל המספרים של {L('Growth Plus')} ושל שני התיקים החודשיים של אותה גרסה נבנו מחדש בקוד נפרד ותאמו. "
    f"ההחלטות שבאו אחר כך ({L('Aggressive 60 / 20 / 20')}, {L('Monthly 60 / 40')}, מינוף {L('1.40')}, פרק העמלות) הורצו מחדש ונבדקו על ידי בלבד, ועברו אישור מנוע; הן לא עברו סוקר בלתי תלוי.</li>"
    f"<li><b>מה הם מצאו:</b> פער אחד: המרווח הדק של התיקים החודשיים מול המדרגה שלהם לא נאמר בעמוד. עכשיו הוא כתוב בכרטיסים, בטבלת המדרגות וב״חשוב לדעת״. תוקנו גם כעשרים ניסוחים ותוויות.</li>"
    f"<li><b>הגרסה הקודמת ({L('v5.1')}):</b> אחרי ההחלטה על {L('40 / 30 / 30')} ל־{L('Growth')} כל המחקר הורץ מחדש, אישור המנוע חזר על עצמו, ושני סוקרים בלתי תלויים בדקו את העמוד: המספרים של {L('Growth')} שוחזרו בקוד עצמאי, ותוקנו ניסוחים שנשארו מגרסת ההון השווה.</li>"
    f"<li><b>שלושת הסבבים הראשונים</b> (למטה) נעשו על גרסת ההון השווה, על ידי סוכנים נפרדים, כל אחד עם קוד משלו. המספרים ברשימות למטה שייכים לגרסה ההיא, והתיק החודשי שמוזכר בהן הוא התיק מ־1 באוקטובר (ארבעה פודים).</li>"
    "</ul>"
    "<ul class=\"plain\">"
    f"<li><b>לפני ההקפאה, התוכנית:</b> שלושה סוקרים חסמו את הטיוטה הראשונה (חסר גילוי של מה שכבר נראה, שער ה־{L('MR')} נשמט, כללים לא מוגדרים). התוכנית נכתבה מחדש ורק אז הוקפאה ({L('70f5c22')}).</li>"
    f"<li><b>הנתונים:</b> שמונה בדיקות: הרצה מחדש של כל הרגליים על הקוד הנוכחי, מיפוי קלט, טריות, סיבתיות. אין הבדל מעבר לעדכון נתונים של {L('Norgate')} ({L('0.0055%')} ליום).</li>"
    f"<li><b>התוצאות:</b> חמישה סוקרים. שניים בנו מחדש את כל מספרי הכותרת מקובצי המקור בקוד משלהם: {L('272')} השוואות, ההבדל הגדול ביותר קטן מ־{L('0.000001')}. "
    f"בדיקת תזמון ודליפה: אין מבט קדימה; שער ה־{L('VIX')} מיושר ליום הקודם ({L('100%')} מהקניות בימי שער פתוח); הרצה שנגמרת ב־2019 משחזרת בדיוק את ההרצה המלאה.</li>"
    "</ul>"
    "<p>אף מספר כותרת לא היה שגוי. מה שתוקן בעקבות הסקירה, כולו בכיוון של פחות אופטימיות:</p>"
    "<ul class=\"plain\">"
    f"<li>היתרון של {L('Growth')} על התיק החודשי הוצג רק בלי עלות נוספת ({L('90%')}). עכשיו הוא מוצג גם ב־{L('+5 bps')} ({L('71%')}) וב־{L('+10 bps')} ({L('42%')}) ולפי תקופה; הסיכום נכתב מחדש.</li>"
    f"<li>קיבולת: פקודות ה־{L('BIL')} של ה־{L('MR')} דיללו את מבחן ה־{L('ETF')}, וסינון ההשתתפות אוחד בין הרגליים. אחרי התיקון: {L('$2.5M')} בפתיחה ו־{L('$5M')} בסגירה (במקום {L('$10M')}), וסינון לפי רגל {L('$3M')} (במקום {L('$34.5M')}).</li>"
    f"<li>חשיפת השיא לנאסד״ק מוצגת גם עם הסחיפה בין איפוסים ({L('1.5×')} ו־{L('2.0×')}), לא רק במשקלי היעד.</li>"
    f"<li>{R}{L('Aggressive')}: נוספה הקריאה בגבול {L('−25%')} (עובר רק עד {L('0.85')} מהיתרון), והחלופה {L('TAA 3x 1N 40 / 30 / 30')} (מאז 5 באוקטובר זה {L('Growth Plus')}).</li>"
    f"<li>בתיק החודשי ״{L('TAA')} מת״ כיבה רק פוד אחד מתוך שניים על אותו מנוע; נוספה העמודה עם שניהם ({L('0.31')}).</li>"
    f"<li>ניסוח המרג׳ין, תוויות חלונות המשבר, אישור המנוע לשלושת המוצרים, ויומן התיקונים לתוכנית (היה ריק).</li>"
    "</ul>"
    f"<p class=\"cap\">מה שנשאר פתוח ולא ניתן לסגור בסימולציה: עלות הביצוע האמיתית של ה־{L('MR')}, והאם המנועים ימשיכו לעבוד. הפריטים שחושבו ולא מוצגים בעמוד רשומים ברשומה האנגלית.</p>")
