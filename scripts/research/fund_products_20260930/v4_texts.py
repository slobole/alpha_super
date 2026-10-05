"""Texts and layout for report v4: statistics and table labels in English, short Hebrew explanations (owner, 2026-10-01)."""

R = "‏"   # RLM: keeps a Hebrew line from starting with a Latin token

LABELS = {
    "C_L": "CORE5 60 / BTAL_QQQ 40", "C_5050": "CORE5 50 / BTAL_QQQ 50", "C_L+DV2_10": "60/40 + 10% DV2 (stocks)", "C_L+HPI_10": "60/40 + 10% HPI",
    "C_N": "CORE5 / BTAL_QQQ / DV2-IND thirds", "C_N6040": "60/40 + 1/3 DV2-IND", "C_L+IND_25": "60/40 + 25% DV2-IND", "C_L+IND_20": "60/40 + 20% DV2-IND",
    "C_DS3": "CORE5 / BTAL_QQQ / downshock thirds", "C_L+DS_20": "60/40 + 20% downshock", "C_L+DS_10": "60/40 + 10% downshock", "C_L+DS_05": "60/40 + 5% downshock",
    "C_T": "CORE5 / BTAL_QQQ / EOM / DV2-IND quarters", "C_T+DS_05": "Target + 5% downshock", "C_T+DS_10": "Target + 10% downshock",
    "C_T6040": "Target, CORE5 30 / BTAL_QQQ 20", "C_EOMDS": "CORE5 / BTAL_QQQ / EOM / downshock quarters", "C_EOM3": "CORE5 / BTAL_QQQ / EOM thirds",
    "G22_launch_alone": "Growth launch, levered", "G22_launch_mix_33": "1/3 growth + 60/40, levered", "G22_launch_mix_50": "1/2 growth + 60/40, levered",
    "G22_launch_mix_67": "2/3 growth + 60/40, levered", "G22_next_mix_33": "1/3 growth + next core, levered", "G22_next_mix_50": "1/2 growth + next core, levered",
    "G22_next_mix_67": "2/3 growth + next core, levered", "G22_target_mix_33": "1/3 growth + target core, levered",
    "G22_target_mix_50": "1/2 growth + target core, levered", "G22_target_mix_67": "2/3 growth + target core, levered",
}

ROWS = {
    # ── DEFENSIVE ──
    "d_launch": {"name": "השקה", "en": "Def: Launch", "chip": "LAUNCH", "chipc": "go", "pick": True, "when": "CORE5 wired", "file": "fund_defensive",
                 "why": "הפשוט ביותר שעומד בכלל גם ב־‎+5 bps‎. רק ETF חודשי. שדרוג ראשון: ‏10% downshock במקום ה־cash."},
    "d_next": {"name": "הצעד הבא", "en": "Def: Next", "chip": "NEXT", "chipc": "alt", "when": "DV2-IND forward test", "file": "fund_defensive_next",
               "why": "מנצח את ההשקה ב־99% מהמסלולים. מותנה במבחן קדימה של DV2-IND."},
    "d_calm": {"name": "הגנתי מאוד", "en": "Def: Very defensive", "chip": "AT LAUNCH", "chipc": "stage", "when": "CORE5 wired", "file": "fund_defensive_calm",
               "why": "ההשקה עם 35% cash. נטו אין כאן אלפא, ולכן כדאי לא לגבות עמלה על ה־cash."},
    "d_rich": {"name": "יותר תשואה", "en": "Def: More return", "chip": "AT LAUNCH", "chipc": "alt", "when": "CORE5 wired", "file": "fund_defensive_plus",
               "why": "‏60/40 עם 20% מתיק הצמיחה ו־20% cash. נקודת קצה, ולכן ציפייה סבירה היא ‏CAGR ~9.2%."},
    "d_target": {"name": "יעד אידיאלי", "en": "Def: Target", "chip": "TARGET · CONDITIONAL", "chipc": "tgt", "when": "EOM + DV2-IND (forward tests)", "file": "fund_defensive_target",
                 "why": "אף שנה שלילית, והיחיד בהגנתי עם אלפא נטו מובהקת. ‏EOM צריך גם נתיב ביצוע סחיר (היום MOC)."},
    "d_ds_upgrade_10": {"name": "השקה + 10% downshock", "en": "Def: Launch + downshock"},
    "d_launch_gated": {"name": "השקה + 9% HPI (מותנה)", "en": "Def: Launch + HPI"},
    "d_calm_next": {"name": "הגנתי מאוד, אחרי הצעד הבא", "en": "Def: Very def. (next)", "dim": True},
    "d_ref_d0": {"name": "60/40 בלי cash (לא עומד בכלל)", "en": "60/40 no cash", "dim": True},
    # ── GROWTH ──
    "g_launch": {"name": "השקה", "en": "Growth: Launch", "chip": "LAUNCH", "chipc": "go", "pick": True, "when": "CORE5 wired", "file": "fund_growth",
                 "why": "ההכרעה מ־30.9. חודשי, ארבעה פודים, עובר ‎$250M‎."},
    "g_plus": {"name": "יותר תשואה, בלי מינוף", "en": "Growth: Plus", "chip": "NO MARGIN", "chipc": "alt", "when": "CORE5 wired", "file": "fund_growth_plus",
               "why": "אותם פודים, יותר TAA. אותו סיכון כמו 22% במינוף, בלי חשבון מרג׳ין."},
    "g_g22_launch": {"name": "‏22% עכשיו, במינוף", "en": "Growth 22%: now", "chip": "MARGIN", "chipc": "stage", "when": "CORE5 + margin",
                     "why": "ההשקה במינוף. לא עדיף על \"יותר תשואה\" בלי מינוף."},
    "g_g22_next": {"name": "‏22% אחרי הצעד הבא", "en": "Growth 22%: next", "chip": "CONDITIONAL", "chipc": "stage", "when": "DV2-IND + margin"},
    "g_g22_target": {"name": "‏22% באותו סיכון", "en": "Growth 22%: target", "chip": "TARGET · CONDITIONAL", "chipc": "tgt", "when": "EOM + DV2-IND + margin",
                     "why": "חצי צמיחה וחצי היעד ההגנתי, במינוף. ‏Max DD כמו ההשקה, ו־22%."},
    "g_g22_maxlev": {"name": "מינוף מרבי על היעד ההגנתי", "en": "Growth 22%: max lev.", "dim": True},
    "g_mr": {"name": "יעד מותנה: היפוך לממוצע", "en": "Growth: MR target", "chip": "CONDITIONAL", "chipc": "tgt", "when": "HPI-RSI rewire + live costs", "file": "fund_growth_mr",
             "why": "‏$10M‎ בלבד, והיתרון נעלם ב־8–10 bps לצד."},
    "g_mr22": {"name": "יעד מותנה במינוף", "en": "Growth: MR levered", "dim": True},
}

META = {
    "rows": ROWS,
    "def_cards": ["d_launch", "d_next", "d_calm", "d_rich", "d_target"],
    "def_menu": ["d_launch", "d_next", "d_calm", "d_rich", "d_target", "#Upgrades and references", "d_ds_upgrade_10", "d_launch_gated", "d_calm_next", "d_ref_d0"],
    "gro_cards": ["g_launch", "g_plus", "g_g22_launch", "g_g22_target", "g_mr"],
    "gro_menu": ["#No leverage", "g_launch", "g_plus", "g_mr", "#Leverage, all at ~22% CAGR", "g_g22_launch", "g_g22_next", "g_g22_target", "g_g22_maxlev", "g_mr22"],
    "frames": ["d_launch", "d_ds_upgrade_10", "d_next", "d_target", "g_launch", "g_plus", "g_g22_launch", "g_g22_target"],
    "crises": ["d_launch", "d_ds_upgrade_10", "d_next", "d_calm", "d_rich", "d_target", "g_launch", "g_plus", "g_g22_target"],
    "years": ["d_launch", "d_next", "d_target", "g_launch", "g_plus", "g_g22_target"],
    "alpha": ["d_launch", "d_ds_upgrade_10", "d_next", "d_calm", "d_rich", "d_target", "g_launch", "g_plus", "g_g22_launch", "g_g22_next", "g_g22_target", "g_g22_maxlev", "g_mr"],
    "chart": [["d_launch", "var(--def)"], ["d_target", "var(--s4)"], ["g_launch", "var(--gro)"], ["g_g22_target", "var(--s5)"]],
}

LI = lambda items: "".join(f"<li>{x}</li>" for x in items)  # noqa: E731

TEXT = {
    "LEDE": "שני מוצרים, ותפריט קצר לכל אחד. הכללים נקבעו מראש והופעלו באופן מכני; סוקר בלתי תלוי עבר על הכול פעמיים, ותיקנתי בעקבותיו.",
    "SUMMARY": LI([
        "<b>הגנתי, השקה:</b> ‏60/40 + 10% cash. ‏<span class=\"ltr\">CAGR 8.2%, Max DD −5.6%</span>. חסר רק CORE5.",
        "<b>שדרוג ראשון:</b> ‏10% downshock במקום ה־cash. ‏<span class=\"ltr\">CAGR 9.0%</span>, מנצח את ההשקה ב־100% מהמסלולים. צריך נתיב לייב.",
        "<b>הגנתי, הצעד הבא והיעד:</b> ‏<span class=\"ltr\">Excess Sharpe 1.26 → 1.61</span>. מותנים במבחן קדימה של DV2-IND ושל EOM.",
        "<b>צמיחה, השקה:</b> ‏<span class=\"ltr\">CAGR 18.1%, Max DD −13.1%</span>. בלי מינוף אפשר עד ‏<span class=\"ltr\">21.4%</span>.",
        "<b>מעל 22%:</b> עכשיו עד ‏<span class=\"ltr\">~26%</span> במינוף ‏<span class=\"ltr\">1.5×</span> (‏<span class=\"ltr\">Max DD ~−20%</span>). ‏28%–30% רק אחרי EOM ו־DV2-IND.",
        "<b>אלפא נטו:</b> מובהקת רק כשהיעד ההגנתי בפנים.",
    ]),
    "DEF_INTRO": "ליבה ועוד חוגת cash. הליבה משתדרגת עם החיווט: ‏CORE5 ו־BTAL_QQQ, אחר כך downshock, ‏DV2-IND, ובסוף EOM.",
    "DEF_CAP": ("כל חמש האפשרויות עומדות בכלל: ‏<span class=\"ltr\">Max DD ≥ −7%, P(DD&lt;−10%) ≤ 10%</span>, גם בעלויות המודל וגם ב־‎+5 bps‎, "
                "ו־2008 ו־2022 לא גרועים מ־‎−1%‎. ‏downshock: מסוכן לבד (‏<span class=\"ltr\">2008 −15.5%</span>), טוב ב־10% בתוך 60/40."),
    "GRO_INTRO": "בלי מינוף: השקה, \"יותר תשואה\", ויעד היפוך לממוצע (מותנה). עם מינוף כל האפשרויות מכוונות ל־22%, וההבדל ביניהן הוא רק הסיכון.",
    "GRO_CAP": ("מינוף יומי, מימון ‏<span class=\"ltr\">DTB3 + 1.5%</span>. ‏<span class=\"ltr\">Reg-T</span> מעל 90% מסומן ⚠ (צריך Portfolio Margin). "
                "הגודל המרבי מחולק במינוף."),
    "A7_INTRO": "בלי מינוף יש תקרה: ‏TAA3x-1N לבדו נותן ‏<span class=\"ltr\">CAGR 27.7%, Max DD −27%</span>. עם מינוף כל מספר אפשרי, ומה שקובע הוא על מה ממנפים.",
    "A7_NOTES": LI([
        "<b>עכשיו:</b> מינוף על ההשקה עדיף על יותר TAA בלי מינוף. התקרה המעשית היא ‏<span class=\"ltr\">CAGR ~24–26%</span>.",
        "<b>אחרי EOM ו־DV2-IND:</b> ‏<span class=\"ltr\">CAGR 28%</span> עם ‏<span class=\"ltr\">Max DD −17%</span>, אבל במינוף ‏<span class=\"ltr\">2.5×</span>, עם Portfolio Margin, ורגיש לעלויות.",
        "<b>יום קריסה:</b> ‏TAA מחזיק TQQQ. בהשקה במינוף ‏<span class=\"ltr\">1.68×</span>, יום של ‏<span class=\"ltr\">−20%</span> בנאסד״ק יכול לעלות כ־‏<span class=\"ltr\">−39%</span>. אירוע כזה לא בתקופת הבדיקה.",
        "<b>נטו ללקוח:</b> ‏2/20 מוריד בערך 6–7 נקודות.",
    ]),
    "ROB_INTRO": "חלופה מחליפה ברירת מחדל רק אם היא עוברת שש בדיקות. מתחת: אותן אפשרויות תחת הנחות אחרות.",
    "ALPHA_TEXT": ("רגרסיה שבועית (Newey-West). נטו מובהק (‏<span class=\"ltr\">t ≥ 2</span>) רק כשהיעד ההגנתי או היפוך לממוצע בפנים. "
                   "מינוף מעלה את ה־t באופן מכני, וכל המספרים בתוך המדגם."),
    "CRISES_CAP": "בכל תא: התשואה לאורך החלון, ובסוגריים הירידה הגרועה בתוכו. לפני 10/2012 ‏TAA ו־BTAL_QQQ הם פרוקסי.",
    "CORR_CAP": "‏TAA3x, ‏TAA3x-1N ו־BTAL_QQQ הם מנוע אחד, שנושא 66%–81% מהסיכון של מוצרי הצמיחה.",
    "DROPPED": LI([
        "<b>‏downshock מעל 20%, או במקום DV2-IND:</b> שובר את רצפת המשברים, או נחות.",
        "<b>‏HPI או DV2 בהשקה:</b> ב־‎+10 bps‎ היתרון כמעט נעלם. מותנה במדידת עלות בלייב.",
        "<b>‏TAA בגרסאות 2x (QLD, ‏SSO):</b> רמת מינוף, לא אסטרטגיה.",
        "<b>ליבה 50/50, הטיות של היעד, ‏60/40 + 20–25% DV2-IND:</b> כולן נחותות מברירות המחדל.",
        "<b>‏35% CAGR:</b> גם במסלול הטוב ביותר זה מינוף ‏<span class=\"ltr\">3.2×</span> וחצי סיכוי לשבור ‏<span class=\"ltr\">−25%</span>.",
    ]),
    "TODO": LI([
        "<b>לאשר את ההשקות:</b> <span class=\"code\">fund_defensive</span> ו־<span class=\"code\">fund_growth</span> (חיווט CORE5).",
        "<b>‏downshock:</b> נתיב לייב, וזה השדרוג הראשון (<span class=\"code\">fund_defensive_ds</span>).",
        "<b>‏DV2-IND:</b> מבחן קדימה והרצה מחדש של ההיסטוריה שלפני 2010.",
        "<b>‏EOM:</b> מבחן קדימה ונתיב ביצוע סחיר.",
        "<b>‏HPI:</b> למדוד עלות ביצוע בלייב.",
        "<b>מינוף מעל 22%:</b> להחליט על הסיכון: ‏<span class=\"ltr\">Max DD</span> שהלקוחות מוכנים לספוג.",
    ]),
    "CAVEATS": ("סימולציה, 2008–2026. לפני 10/2012 ‏TAA ו־BTAL_QQQ הם פרוקסי, ו־DV2-IND לפני 2010 עדיין לא תקף. "
                "זה לא מבחן עיוור: הכללים נכתבו אחרי ששלבים קודמים נראו, ולכן הם נקבעו מראש והופעלו מכנית, וכל תיקון מתועד במפרט."),
}
