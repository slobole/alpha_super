"""Detailed evidence appendix and local knowledge record; no new experiments."""
from __future__ import annotations
from datetime import datetime, timezone
import html
import json
import re
import pandas as pd
import markdown
from scripts.research.portfolio_family_20260923.freeze import STUDY_PATH
from scripts.research.portfolio_family_20260923.report import CORE_LIST, NAME_DICT, percent_str, table_str


def main() -> None:
    table_path=STUDY_PATH/"tables"
    spec_dict=json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    metadata_dict=json.loads((STUDY_PATH/"effective_source_metadata.json").read_text(encoding="utf-8"))
    catalog_list=json.loads((STUDY_PATH/"catalog_rules.json").read_text(encoding="utf-8"))
    defensive_list=json.loads((STUDY_PATH/"defensive_rules.json").read_text(encoding="utf-8"))
    catalog_dict={record_dict["strategy_import"].split(":")[0].split(".")[-1]:record_dict for record_dict in catalog_list}
    for record_dict in defensive_list:
        source_id_str=record_dict["import"].split(":")[0].split(".")[-1]
        catalog_dict[source_id_str]={**catalog_dict[source_id_str],"audit_status":"source_rules_verified", "defensive_details":record_dict}
    metric_df=pd.read_csv(table_path/"portfolio_metrics.csv")
    primary_df=metric_df.query("scenario=='common_account' and rebalance=='annual_fixed'").set_index("candidate_id")
    standalone_df=pd.read_csv(table_path/"all25_common_metrics.csv").query("scenario=='common_account'").set_index("source_id")
    period_df=pd.read_csv(table_path/"subperiod_metrics.csv")
    gate_df=pd.read_csv(table_path/"frozen_gate_results.csv")
    inventory_rows=[]
    for source_id_str,record_dict in catalog_dict.items():
        source_dict=metadata_dict[source_id_str];row_series=standalone_df.loc[source_id_str]
        cadence_str=record_dict.get("cadence",record_dict.get("defensive_details",{}).get("actual_order_cadence","ראו כללים"))
        inventory_rows.append([record_dict["alias"],record_dict["friendly_name_he"],record_dict["tier"],percent_str(row_series.cagr),percent_str(row_series.max_drawdown),f'{source_dict["slippage_per_side_float"]*10000:g}',f'{source_dict["commission_per_share_float"]:g}/{source_dict["commission_minimum_float"]:g}'])
    inventory_table_str=table_str(["כינוי","מנגנון / שם","שלב","תשואה שנתית","ירידה מרבית","החלקה בכל צד, נ״ב","עמלה $/מניה ומינימום"],inventory_rows)
    period_rows=[]
    for candidate_str in CORE_LIST:
        for period_int in (1,2,3):
            row_series=period_df.query("scenario=='common_account' and rebalance=='annual_fixed' and candidate_id==@candidate_str and period==@period_int").iloc[0]
            period_rows.append([NAME_DICT[candidate_str],period_int,percent_str(row_series.cagr),percent_str(row_series.volatility),percent_str(row_series.max_drawdown),percent_str(row_series.es5_loss,2)])
    period_table_str=table_str(["תיק","תקופה","תשואה שנתית","תנודתיות","ירידה מרבית","הפסד ממוצע ב־5% הימים הגרועים"],period_rows)
    exposure_rows=[]
    for candidate_str in CORE_LIST:
        row_series=primary_df.loc[candidate_str]
        exposure_rows.append([NAME_DICT[candidate_str],percent_str(row_series.average_gross),percent_str(row_series.maximum_gross),percent_str(row_series.average_short),percent_str(row_series.maximum_short),percent_str(row_series.es5_loss,2),percent_str(row_series.worst_month),percent_str(row_series.worst_rolling252)])
    exposure_table_str=table_str(["תיק","חשיפה ברוטו ממוצעת","ברוטו מרבי","שורט ממוצע","שורט מרבי","הפסד יום קיצון ממוצע","חודש גרוע","252 ימים גרועים"],exposure_rows)
    tail_df=pd.read_csv(table_path/"exposures_primary_pair_diagnostics.csv.gz")
    pair_df=tail_df.query("subset=='full_primary'")
    alias_dict={source_id_str:record_dict["alias"] for source_id_str,record_dict in catalog_dict.items()}
    correlation_table_str=table_str(["זוג","מתאם","שיעור ימים שבהם שניהם הפסידו"],[[f'{alias_dict[row_obj.left_source]} / {alias_dict[row_obj.right_source]}',f'{row_obj.correlation:.2f}',percent_str(row_obj.joint_loss_fraction)] for row_obj in pair_df.itertuples()])
    native_df=pd.read_csv(table_path/"native_replay_metrics.csv")
    native_table_str=table_str(["הרצה","הון התחלתי בדולר","תשואה שנתית","תנודתיות","ירידה מרבית"],[[row_obj.case_id,f'{row_obj.capital:,.0f}',percent_str(row_obj.cagr,3),percent_str(row_obj.volatility,3),percent_str(row_obj.max_drawdown,3)] for row_obj in native_df.itertuples()])
    allocation_rows=[]
    for candidate_dict in spec_dict["portfolio"]["candidates"]:
        allocation_rows.append([candidate_dict["candidate_id"],", ".join(f'{alias_dict.get(source_id_str,"BIL")} {weight_float*100:g}%' for source_id_str,weight_float in candidate_dict["weights"].items())])
    allocation_table_str=table_str(["מזהה מחקר","הרכב הון מדויק"],allocation_rows)
    detail_parts=[]
    for source_id_str,record_dict in catalog_dict.items():
        source_dict=metadata_dict[source_id_str]
        detail_parts.append(f'### {record_dict["friendly_name_he"]} / {record_dict["alias"]}\n\n`{source_id_str}`\n\n'
            +f'כיסוי הריצה המקורית: {source_dict["actual_start_date_str"]} עד {source_dict["actual_end_date_str"]}. הון מקור: ${source_dict["native_capital_float"]:,.0f}.\n\n'
            +'```json\n'+json.dumps(record_dict,ensure_ascii=False,indent=2)+'\n```\n')
    report_str=rf'''# נספח ראיות — משפחת תיקי מאקרו וצמיחה

## איך לקרוא את הנספח

[חזרה לדוח הראשי](REPORT.html). ההכרעות נמצאות בדוח הראשי. כאן נמצאים ההרכבים המדויקים, כל 25 האסטרטגיות, עלויות, תקופות משנה, חשיפות, נוסחאות ומגבלות. כל התוצאות הן מחקר היסטורי. שום מדיניות תיק או אסטרטגיה בסביבת המסחר לא שונתה.

## היקום וכלל ההכללה

הכלל היה כל אסטרטגיה הרשומה PM_READY או WIRED: 25 בסך הכול, 16 ו־9 בהתאמה. מעמד זה מעיד על חיבור ובדיקות מערכת, לא על יתרון השקעה. הטבלה הבאה היא בתרחיש הבסיס common_account, עם הנחת ניכוי 25% מדיבידנד קנוי ומימון 5%, בחלון המשותף **5.4.2019–31.7.2026**: סגירת 4.4.2019 היא עוגן, אחריה 1,840 תשואות. אסור להשוות ישירות את המספרים כאן לטבלה הראשית המתחילה ב־2012.

{inventory_table_str}

לא נבחרו רק בעלי התשואה הגבוהה. ארבע אסטרטגיות הצמיחה והמאקרו נקבעו מראש כנציגי מנגנונים, ונבדקו גם כל היתר לבד, חלופות TAA, חלופות הגנה והתיקים הישנים. מדיניות השימוש בכל אסטרטגיה מופיעה ב[מפת התפקידים](verification/eligible_universe_decisions.md). העדר משקל בתיק הראשי אינו הכרזה שאסטרטגיה גרועה.

נתוני שוק מקוריים מריצות שונות מגיעים בתאריכי עדכון שונים. לכל קובץ מקור וייצוא יש חותמת SHA-256, אך ברוב הריצות חסרה חותמת של הקוד ושל ספק הנתונים בזמן יצירת הריצה. אימות זהות הקבצים כיום אינו שחזור של גרסת העבר.

## מה בדיוק נספר במחקר

נקבעו 36 הרכבים × שתי מדיניות איזון × שלושה תרחישי עלות = 216 תאים. בנוסף 25 אסטרטגיות × שלושה תרחישים = 75; שני מדדי ייחוס; שש הרצות CORE5 ועוד הרצת מקור אחת להוכחת שוויון = שבע. סך הכול **300 תאי חישוב**, מתחת לתקרה 320. חלונות משנה, מתאמים, אחזקות והפעלת בדיקות אינם הרכבים חדשים.

לאחר השלב הזה נוסף סבב הסתגלות אחד, H10, עם השערה אחת ושישה הרכבים × שלוש עלויות באיזון שנתי בלבד: עוד 18, כלומר **318 תאים סופיים ו־42 הרכבים שונים**. תשע הרצות השוואה חוזרות של תיקים מקוריים שימשו בדיקת שוויון ואינן אפשרויות השקעה נוספות. לא נוסו פרמטרים או משקולות נוספים אחרי התוצאות. תקרת 320 המקורית נשמרה.

ל־CORE5 תועדו במחקר קודם 441 מסלולים נומינליים, ול־Portfolio Foundry תועדו 64 בדיקות. יש גם מחקרי המשך והיסטוריית Ladder. מידת החפיפה אינה ידועה, ולכן אין לחבר אותם כאילו היו ניסויים בלתי תלויים. שום חלון היסטורי כאן אינו מדגם חדש שלא נצפה. 19 בדיקות Holm מתקנות רק את המשפחה הנוכחית.

התוכנית המקורית נשארה ללא שינוי: [כללי המחקר](research_spec_frozen.json). [תוספת לפני התוצאות](pre_result_amendment_01.json) הבהירה חשבונאות סוף חלון והרצות CORE5. תיקוני המימוש לאחר הביקורת עסקו בעלויות המוצגות ובמתאם שאינו מוגדר במדגם קצר; הם לא שינו הרכבים, תקופות או תשואות תיק. בדיקות החשיפה הוגדרו בנפרד כתיאור של המסלולים הקיימים, ללא חיפוש משקל חדש.

## המשקולות המדויקות

{allocation_table_str}

משקולות ה־Ladder הן קריאה של קובצי התיקים הקיימים, בהדמיית יחידות בסכום ייחוס של מיליון דולר. אלה אינם שחזור של החשבון המקורי או של כללי גודל הפקודה שלו. הקוד החדש אינו כותב לקובצי ה־YAML.

## תקופות משנה ומשברים

תקופה 1: 2.10.2012–30.12.2016. תקופה 2: 3.1.2017–31.12.2020. תקופה 3: 4.1.2021–31.7.2026. כולן מוכרות מהמחקר הקודם; החלוקה היא בדיקת יציבות, לא מבחן מחוץ למדגם.

{period_table_str}

כל ארבע מדרגות הסיכון עברו את הסף גם בעלויות בסיס וגם בעלויות מחמירות, ובכל שלוש התקופות. אין מכאן הבטחה לסדר מושלם של ירידה משיא: בתרחיש המחמיר הירידה המרבית של 75% מאקרו הייתה מעט קטנה מזו של 100% מאקרו, 5.94% מול 6.00%. בחלק מתקופות המשנה התרחשו היפוכים נוספים. הדרישה המקורית עסקה בתנודתיות ובהפסד יומי בזנב, לא בדירוג מושלם של כל מדד.

משברים שנקבעו מראש: 18.8.2015–11.2.2016; 2–9.2.2018; 1.10–24.12.2018; 20.2–23.3.2020; 3.1–12.10.2022; 3.2–30.4.2025. [כל התוצאות](tables/crisis_metrics.csv). משברי 2008 ו־2011 נבדקו רק לאסטרטגיות שלהן כיסוי ממשי; אין הארכת ETF לפני תחילת נתוניו. [כיסוי ותוצאות ארוכות](tables/long_history_metrics.csv).

ל־CORE5: בחלון המשבר 15.9.2008–9.3.2009 תשואה כוללת של 8.20% וירידה משיא של 2.45%; ב־1.8–4.10.2011 תשואה של ‎−2.26% וירידה של 4.81%. אלה חלונות שנבחרו מראש מתוך היסטוריה מוכרת. לא מוצגת תשואה שנתית מחלון משבר קצר כאילו הייתה קצב בר־קיימא.

## חשיפה, מינוף וריכוזיות

{exposure_table_str}

ברוטו הוא סכום הערכים המוחלטים של אחזקות ניירות ערך חלקי ההון; מזומן אינו נספר. שורט נספר בנפרד. CORE5 מכוון ל־100% קניות ועוד שורט DBC עד 10%, ולכן התווית ״הגנתי״ אינה ״ללא שורט״. פערי פתיחה וסטייה במשקולות יכולים להביא לחשיפה בפועל גבוהה מהיעד. תיקים הכוללים TQQQ או QLD דורשים גם מבט על המינוף שבתוך הקרן; שווי הקרן בתיק אינו החשיפה הכלכלית שלה.

חשיפת התיקים מחושבת ממשקל האסטרטגיה בסוף היום, כפול האחזקות המקוריות שלה באותו סוף יום. תחת עלויות מתוקנות זו הערכה על אחזקות היסטוריות, לא מניות שנרכשו מחדש אחרי העלות. ריכוזיות נספרת ברמת טיקר; אין המצאה של אחזקות פנימיות בתוך SPY, QQQ או BTAL. אין קיזוז סמוי של שורט וקנייה בין אסטרטגיות. ל־Trinity חסרה מטריצת אחזקות יומית, ולכן ידועים רק סך ההשקעה והמזומן; אין לה מטריצת שמות מומצאת.

[ריכוזיות יומית](tables/exposures_core_concentration_daily.csv.gz), [סיכום ריכוזיות](tables/exposures_core_concentration_summary.csv.gz), [פירוט אחזקות](tables/exposures_core_assets.csv.gz), [תרומות לסיכון](tables/risk_contributions.csv).

## האם באמת מתקבל פיזור

{correlation_table_str}

NDX/Mosaic: מתאם יומי 0.721 בכל החלון; 0.847, 0.820 ו־0.598 בשלוש התקופות. ב־174 ימי SPY הגרועים ביותר המתאם היה 0.834, וב־59.2% מהם שתיהן הפסידו. בימים אלה, כאשר NDX הפסידה, Mosaic הפסידה ב־94.5% מהמקרים. אלה תיאורים מותנים של אותו עבר, לא תחזית.

חפיפת השמות הממוצעת ביחידות NAV הייתה 12.74% על כל 3,476 הימים. לאחר נרמול רק להון המושקע בקניות, ורק ב־2,914 ימים שבהם שתיהן השקיעו, החפיפה הייתה 15.33%. ממוצע החפיפה אינו אחוז המניות המשותפות ואינו אחוז מ־Ladder 4. הנוסחה היא:

$$
O_t=\sum_a\min(\max(w_{{NDX,a,t}},0),\max(w_{{Mosaic,a,t}},0))
$$

לכל נייר לוקחים את המשקל החיובי הנמוך מבין שתי האסטרטגיות ומסכמים. [חפיפות](tables/exposures_ndx_mosaic_overlap_summary.csv.gz), [שמות משותפים](tables/exposures_ndx_mosaic_shared_assets.csv.gz), [תלות מתגלגלת](tables/exposures_ndx_mosaic_rolling126.csv.gz). [מתאמי כל 25 האסטרטגיות](tables/exposures_all25_correlation_common_account.csv.gz) מתייחסים לחלון הקצר שמתחיל באפריל 2019.

תרומת רכיב לשונות מחושבת כ־Cov(תרומתו היומית, תשואת התיק) / Var(תשואת התיק). תרומה להפסדי זנב מחושבת מסכום תרומת הרכיב ב־5% ימי התיק הגרועים, חלקי סכום תשואת התיק באותם ימים. התרומות מסתכמות ל־100% לפי בניית החישוב, אך אינן אחוזי כסף ואינן אחוזי סטיית תקן.

## חשבונאות ועלויות

תשואות המקור הן NAV לאחר העלויות המקוריות, עם דיבידנדים שנזקפו במזומן. מחירי ביצוע ושערוך הם CAPITALSPECIAL; לכל מקור מתועד בסיס האות, ולעתים הוא TOTALRETURN. לא מוסיפים שוב דיבידנד לסדרה שכבר כוללת אותו. עלות מקור אינה אחידה: יש אסטרטגיות עם 1, 2.5, 5 או 10 נקודות בסיס לצד, ועמלות שונות; הטבלה למעלה והקטלוג למטה מפרטים.

בסיס: ניכוי 25% מדיבידנד חיובי על קניות בלבד; שורט מחויב במלוא הדיבידנד שהוא חייב. היכן שכבר נוכה 25%, אין חיוב נוסף. זו הנחת תרחיש, לא פסיקת מס. ריבית על מזומן חיובי נשארת כפי שהיא במקור — לרוב אפס; BIL מרוויח דרך אחזקת הקרן ודיבידנדיה. Fixed Income מזכה ריבית DGS3MO במקור; אין זיכוי שני. ריבית אג״ח/מחיר הקרן אינה זהה לריבית שמקבל חשבון ברוקר על מזומן פנוי.

בסיס חוב למימון הוא max(0, ביטחונות לשורט פחות מזומן). בבסיס נגבים 5% לשנה; במחמיר 8%. משתמשים בביטחונות המעוגלים בפועל כשנשמרו; אחרת ב־102% משווי השורט כתחליף מפורש. מכפילים בימי לוח עד יום המסחר הבא ומחלקים ב־360. המימון נפרד מדמי השאלת ניירות. במחמיר דמי ההשאלה עולים ל־5%, ו־10 נ״ב נוספות מוכפלות בסך שווי המסחר המוחלט בכל יום. אין חיוב מימון מעבר לסוף חלון ההערכה; עמלת מקור שהקדימה תקופה עתידית אחרי הסוף מוחזרת רק בתרחישים המתוקנים. עלות מקור בגין היום עצמו נשארת.

$$
r^*_t=r^{{native}}_t-\frac{{\Delta tax_t+funding_t+extra\_slippage_t+\Delta borrow_t}}{{NAV_{{t-1}}}}
$$

זהו תיקון על אותן אחזקות ופקודות, ולא הרצה מחדש של כל 25 האסטרטגיות. העלות עשויה לשנות קניות עתידיות במציאות. עמודות העלות השנתית הן סכום רכיבי עלות יחסית להון, בקצב 252 ימים; אין לסכמן כאילו היו הפער המדויק ב־CAGR. עמלת מקור ושחיקת מחיר מקור כבר בתוך NAV. הניכוי המקורי מדיבידנד וההחלקה המקורית אינם מפורקים כולם לעמודות נפרדות.

מחזור שנתי = 252 כפול ממוצע סכום הערכים המוחלטים של עסקאות ביום חלקי NAV קודם, עם משקל ההתחלה של כל שרוול. עלות האיזון החיצונית: 0.1% מההון בהקצאה הראשונה, ובהמשך 0.1% כפול סכום השינויים המוחלטים במשקולות בתחילת שנה. אין עלות מכירה דמיונית בסוף המחקר, כי הסיום הוא שערוך, לא מימוש.

## תזמון וגבולות הסימולציה

```text
[Native observed data through Close_T]
                 |
       *** CRITICAL *** only information known by T
                 v
[Unchanged signal + prior-close whole-share sizing]
                 | Open_(T+1), source-specific exceptions documented
                 v
[Native orders / dividends / borrow / daily NAV]
                 |
[Saved-unit portfolio + prior-close annual reset]
                 |
[Common dates / costs / risk / frozen comparisons]
```

Flow הוא חריג מתועד: מסחר בסגירה בשלושה מועדים מתוזמנים בחודש, עם אות מוקדם יותר. אין להכניסו אוטומטית למסלול ״פעם בחודש בפתיחה הבאה״. Crisis מתכנן יעדים חודשיים אבל בודק סטייה מעל 2% מדי יום. Trinity יכול להתאזן לפי סטיית יעד תנודתיות גם באמצע חודש. CORE5 יכול לסחור עם שינוי מצב אחד משרווליו. התדירות נקבעת לפי הפקודות האפשריות, לא לפי שם המשפחה.

הסדרות המקוריות חושבו על 100 אלף דולר, עם מניות שלמות ועמלת מינימום; הכפלת NAV למיליון דולר אינה מחדש את אותן החלטות. שרוולים של 62.5 אלף דולר עשויים להיות שונים מהכפלה יחסית של מקור 100 אלף דולר. איזון ההון בין שרוולים בסגירה קודמת והחלת התשואה היומית הבאה אינו ביצוע פקודות פיזי במחיר פתיחה. לכן נדרשת הרצת חשבון מאוחד לפני שימוש תפעולי, עם שערי פתיחה, עמלות, תזמון, עיגול, מזומן, קיזוזים מוצהרים ואי־השלמת פקודות.

## CORE5 — בדיקת מנוע מלאה

{native_table_str}

עוגן מזומן 28.9.2012, פקודות ראשונות 1.10.2012, סיום 31.7.2026: 3,478 שוויי סגירה ו־3,477 תשואות. החלון שונה ביום מהטבלה הראשית, שבה קונים יחידות של אסטרטגיה שכבר פועלת בסגירת 1.10.2012. נשמר כל היסטוריית החימום המקורית. התוצאות בטבלה אינן הוכחת התאמה אחד־לאחד לסדרה הראשית.

שש הרצות התרחיש שמרו על האותות המקוריים, אבל עלות המימון ירדה מהמזומן והשפיעה על גודל הקנייה הבא. הרצת מקור שביעית ללא תוספת מימון הוכיחה שוויון מדויק ב־NAV, פקודות וכלכלת העסקה, דמי השאלה, אחזקות ויעדים; מזהי פקודות נבדקו לפי הסדר היחסי משום שהמונה המוחלט הוא גלובלי לתהליך. נבדקו זהות NAV, חיובי דיבידנד, נוסחת מימון בכל יום ואפס חיוב אחרי יום הסיום. כל ההרצות כוללות הנחת ניכוי 25% מדיבידנד ארוך.

[הגדרה נעולה](native_replay/spec.json), [חותמות והוכחת שוויון](native_replay/replay_manifest.json), [ביקורת בלתי תלויה](verification/core_review.md).

## שיטות המדידה והבדיקות

CAGR = (מכפלת 1+r) בחזקת 252/N פחות 1. תנודתיות = סטיית תקן מדגמית של התשואה היומית כפול שורש 252. שארפ מול אפס = ממוצע יומי ×252 / תנודתיות. שארפ מעבר ל־BIL משתמש בתשואות r−r_BIL ובסטיית התקן שלהן. ירידה משיא כוללת עוגן 1 לפני התשואה הראשונה, כך שהפסד כניסה אינו נמחק. חודש הוא חודש קלנדרי, והחודש הראשון והאחרון עשויים להיות חלקיים. 252 ימים הם חלון מסחר נגרר; הוא אינו בדיוק שנה קלנדרית.

ES5 הוא מינוס ממוצע התשואות הנמוכות או שוות לאחוזון 5 היומי, לא תחזית להפסד מקסימלי. בטא היא Cov(r_p,r_SPY)/Var(r_SPY). מתאם חודשי מחושב לאחר צבירה עצמאית של התשואות בכל חודש, לא כממוצע מתאמים יומיים. מתאם מתגלגל משתמש רק ב־126 התצפיות האחרונות; כאן הוא כלי תיאור ולא אות לסחר.

דגימת אי־ודאות: 2,000 חזרות, בלוקים מעגליים בני 63 ימים, seed=20260923, אותם ימים לכל זוג תיקים. הסטטיסטי הוא ממוצע הפרש יומי מוכפל ב־252. ערך p חד־צדדי מתפלגות ממורכזת עם תיקון plus-one; רווח 95% מאחוזוני התפלגות הממוצעים. Holm על 19 ההשוואות בלבד. רווחי הביטחון הרגילים אינם רווחים סימולטניים מתוקנים לכל ההיסטוריה. [כל הערכים](tables/paired_bootstrap.csv).

רף הגנה: ירידה משיא קטנה בלפחות 1 נקודת אחוז, ES5 קטן בלפחות 5%, ויתור על עד 0.5 נקודת אחוז CAGR; בכל החלון ובשתי תקופות משנה לפחות, בשני תרחישי העלות. אף תוספת הגנה לא עברה. רף CORE מול BIL: יתרון CAGR של לפחות 0.5 נקודת אחוז, לא יותר מ־10% החמרה ב־ES5 ולא יותר מנקודת אחוז החמרה בירידה; אותם תנאי תקופות ועלויות. רק מדרגת 25% עברה בשני התרחישים. [בדיקות הסף](tables/frozen_gate_results.csv).

## בדיקת ההמשך הענפית — H10

ההשערה נולדה לאחר קריאת תוצאות המקור וכל 25 האסטרטגיות: החלפת HPI במנגנון קניית ירידות ב־11 קרנות ענפיות עשויה להפחית תלות במניות בודדות ובפקודות מתחרות. לא הוכח כאן שיפור בביצוע פקודות. כל המשקולות האחרות נשארו זהות בתיקי 75%, 50% ו־25% מאקרו; משקל HPI שהוחלף היה 6.25%, 12.5% או 18.75%. לכל החלפה נבדק ביקורת זהה עם BIL במקום הרכיב, כדי להפריד בין תרומה של האות לבין הפחתת סיכון באמצעות מזומן.

התוכנית ננעלה ב־23.9.2026 בשעה 13:24:09 UTC לפני חישוב תשואות ההרכבים החדשים. אותם תאריכים, תקופות משנה, עלויות ואיזון שימשו גם כאן. נדרש מול התיק המקורי: אובדן CAGR עד 0.5 נקודת אחוז, ES5 נמוך בלפחות 5%, וירידה משיא שאינה גרועה יותר. מול BIL נדרש יתרון CAGR של 0.5 נקודת אחוז, לכל היותר 10% החמרה ב־ES5 ונקודת אחוז בירידה. כל תנאי נדרש יחד בחלון המלא ובשתי תקופות לפחות, גם בבסיס וגם במחמיר. לא שונו הספים לאחר התוצאה.

ההחלפות צמצמו סיכון, אבל **אף אחת משלוש רמות המאקרו לא עברה את מלוא התנאים**. בבסיס ויתור התשואה מול המקור היה 0.57, 1.14 ו־1.70 נקודות אחוז לשנה. בתיק 50% מאקרו הירידה המרבית השתפרה מ־9.00% ל־7.69%, אך CAGR ירד מ־13.46% ל־12.32%. מול ביקורת BIL באותו תיק התקבלו 11.52% וירידה של 7.03%. לכן ההחלפה מציגה עסקה בין תשואה לסיכון, ולא שיפור שמחליף אוטומטית את המודל.

שש השוואות ממוצע מזווגות נבדקו בנפרד, באותן 2,000 דגימות בלוקים ובתיקון Holm שש־השוואות. שלוש ההשוואות הענפיות מול BIL נתנו p מתוקן של כ־0.003; מול המקור p=1. זה אינו מתקן את עצם בחירת ההשערה לאחר שראינו את העבר, ואינו מחליף את תנאי התועלת הכלכלית שלא עברו. ההכרעה היא לשמור את חלופת הענפים כרעיון לריכוך עם מחיר, בלי עוד כוונון במשקולות ובלי המלצת הקצאה חדשה.

[תוכנית המשך קפואה](adaptive_sector_spec.json), [18 התאים ותשע בדיקות השוויון](tables/adaptive_sector_metrics.csv), [פירוט ספים](tables/adaptive_sector_gate_detail.csv), [שלוש החלטות](tables/adaptive_sector_decisions.csv), [בדיקות ממוצע](tables/adaptive_sector_paired_bootstrap.csv), [יומן התפקידים המשלים](verification/eligible_universe_decisions_h10_addendum.md).

## קיבולת, הטיות ומה לא הוכח

אין כאן בדיקת קיבולת לפי ADV, השתתפות במכרז פתיחה, ספר פקודות, השאלת DBC זמינה או החזרת השאלה כפויה. אין סף הון ״נוח״ מוכח. בדיקת 250 אלף–מיליון דולר ל־CORE5 מאשרת רגישות לעיגול ועמלות במנוע, לא נזילות אמיתית; היא אינה מכסה את שרוולי המניות.

היסטוריית Norgate משמרת חברות היסטורית במדדים, אבל לא שוחזרו כאן כל חותמות ההורדה המקוריות ולא נבדקה מחדש כל רשומת דליסטינג. אותות המתבססים על FRED ונתוני מאקרו עדכניים אינם בהכרח מידע זמין בזמן אמת במועד ההיסטורי. לתיקון עלויות אין טענה לביצוע סיבתי חדש. סיכוני בחירת אסטרטגיות, שווקים חריגים, מילוי מחירי מקור ופרסום מאקרו נשארים ברמת אמינות המקורות שנבדקו. אין לחפש באותו עבר עוד משקולות עד שיתקבל מוצר שנראה מושלם.

העדיפות הבאה: לסגור תקציב סיכון ומטבע עם הלקוח, לבחור מבנה קפוא, לשחזר ביצוע פיזי באותו הון ובאותו מקור נתונים, לבדוק נזילות והשאלות, ואז למדוד קדימה עם קריטריוני כישלון שנקבעו מראש. איסוף ראיות עתידיות נפרד מהפעלה חיה או הקצאת הון.

## מקורות ושחזור

[מקורות וחותמות](source_manifest.json), [תיקוני זהות מדד וריבית מזומן](source_audit_addendum.json), [מדדי ייחוס](benchmarks/benchmark_manifest.json), [תוכנית החשיפות](exposures_spec.json), [מחברת שבוצעה](decision_notebook.ipynb), [מצב מחקר](research_state.json), [יומן ניסויים](experiment_ledger.jsonl), [ביקורת החישוב](verification/analysis_review.md), [קובצי החבילה](run_manifest.json).

המתודולוגיה עולה בקנה אחד עם הצורך להגדיר מטרות ואילוצים לפני בניית תיק, ועם זהירות ממובהקות לאחר חיפוש: [CFA — תכנון תיק](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/basics-of-portfolio-planning-and-construction), [Bailey & López de Prado — Deflated Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). במחקר הזה לא חושב DSR; בוצע Holm למשפחת ההשוואות המקומית, והטיית החיפוש ההיסטורי נשארת מגבלה גלויה.

## קטלוג הכללים שנקראו מהקוד

הקטלוג הבא שומר את הנוסחאות, היקומים, התזמון, העלויות והפניות לקוד. הפרטים הטכניים באנגלית נשמרים כדי למנוע שינוי משמעות בתרגום. מפת המקורות המתוקנת גוברת על תווית מדד ישנה בתוך ריצה שמורה.

'''+"\n".join(detail_parts)
    report_str += "\n\nAudit concepts: source, timing, search, holdout, failure, artifact. No untouched historical holdout exists.\n"
    (STUDY_PATH/"REPORT_FULL.md").write_text(report_str,encoding="utf-8")
    # Collapsible source rules keep the evidence appendix navigable.
    report_html_str=markdown.markdown(report_str,extensions=["tables","fenced_code"])
    report_html_str=re.sub(r"<h3>(.*?)</h3>(.*?)(?=<h3>|\Z)",r"<details><summary>\1</summary>\2</details>",report_html_str,flags=re.S)
    for formula_str in ("Overlap_t = Sum_a min(max(w_NDX,a,t, 0), max(w_Mosaic,a,t, 0))",
                        "Adjusted return_t = Native return_t - (incremental tax + funding + extra slippage + incremental borrow)_t / NAV_(t-1)"):
        report_html_str=re.sub(r"<p>\$\$.*?\$\$</p>","<pre>"+html.escape(formula_str)+"</pre>",report_html_str,count=1,flags=re.S)
    report_html_str=re.sub(r"<td>(.*?)</td>",lambda match_obj:
        '<td dir="ltr">'+match_obj.group(1)+'</td>' if not re.search(r"[\u0590-\u05ff]",match_obj.group(1)) else match_obj.group(0),report_html_str,flags=re.S)
    appendix_html_str='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>נספח ראיות — תיקי מאקרו וצמיחה</title><style>body{font:16px/1.8 'Segoe UI',Arial,sans-serif;color:#18384a;background:#f4f7f8;margin:0}main{max-width:1180px;margin:24px auto;background:white;padding:40px;border-radius:14px}h1,h2{color:#165969}h2{border-top:1px solid #cadde3;margin-top:42px;padding-top:16px}table{width:100%;font-size:12px;border-collapse:collapse;display:block;overflow:auto}th,td{border-bottom:1px solid #d5e3e8;padding:9px;text-align:right}th{background:#e8f1f4}tr:nth-child(even){background:#f8fafb}a{color:#007c88}pre{direction:ltr;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:16px;font:12px/1.6 Consolas,monospace}code{direction:ltr;unicode-bidi:isolate;overflow-wrap:anywhere}details{border:1px solid #d2e0e5;margin:14px 0;padding:12px;border-radius:7px}summary{cursor:pointer;font-weight:bold;color:#15596b}@media(max-width:700px){main{margin:8px;padding:20px}table{font-size:11px}}</style><main>'''+report_html_str+"</main></html>"
    (STUDY_PATH/"REPORT_FULL.html").write_text(appendix_html_str,encoding="utf-8")
    state_dict=json.loads((STUDY_PATH/"research_state.json").read_text(encoding="utf-8"))
    row_series=primary_df.loc["CORE_050"]
    record_dict={"schema_version":"quant-research-knowledge-v1","study_id":"portfolio_family_20260923",
        "title":"CORE5-centered explanatory portfolio families","created_at":datetime.now(timezone.utc).isoformat(),
        "research_status":"forward_hypothesis","disposition":"promising_component","replication_outcome":"not_reproducible",
        "signal_family":"portfolio_construction","objective":spec_dict["objective"],
        "verdict":"Keep a simple CORE5 100/75/50/25 family as transparent research models; no precise optimized weights or client deployment claim.",
        "verdicts":{"source_replication":"Historical run code/vintage cannot be reproduced completely; original files verified. Current native CORE5 zero-funding parity is exact.","predictive_value":"No new predictive signal and no untouched validation sample.","economic_value":"Risk order survives both cost scenarios and all3 partitions; no defensive satellite passes;25%CORE inL4 passes the local CORE-vs-BIL gate.","promotion":"At most a forward hypothesis; physical combined-book replay, account suitability and prospective evidence required."},
        "universes":["25 registered WIRED/PM_READY strategies; native PIT stock universes and observed ETFs"],
        "decision_timing":"Native Close_T; outer prior-close annual reset","fill_timing":"Native Open_T+1 except documented Flow MOC; synthetic outer units, not physical execution",
        "timing_attribution":{"status":"not_applicable","reason":"No new close-derived signal; native timing preserved. Outer execution remains a disclosed proxy."},
        "primary_cost_layer":"central_research","primary_metrics":{"period":"2012-10-02..2026-07-31;3476returns","universe":"CORE_050 illustrative model, not selected winner","cost_layer":"common_account/annual_fixed","CAGR":float(row_series.cagr),"annualized_volatility":float(row_series.volatility),"Sharpe":float(row_series.sharpe_excess_bil),"Sharpe_definition":"mean excess BIL / std excess BIL annualized","maximum_drawdown":float(row_series.max_drawdown),"turnover":float(row_series.annual_turnover)},
        "feature_findings":[{"feature":"CORE5 capital fraction","role":"portfolio_construction","direction":"More CORE reduces volatility and ES5","status":"descriptive_forward_hypothesis","effect_size":"8/8adjacency-scenario gates passed; all3subperiods","period_consistency":"3/3seenperiods bothscenarios","corrected_significance":"Risk gates descriptive; no independent holdout","economic_mechanism":"Macro trend and bills diversify equity momentum/rebound budgets","recommended_action":"Keep frozen coarse family; no fine tuning"}],
        "cost_capacity":{"paper_like_round_trip_bps":"heterogeneous native2to20bps plus commissions; see catalog","central_research_round_trip_bps":"same native transactions plus25%long dividend assumption and5%debt funding","conservative_survival_round_trip_bps":"native plus20bps roundtrip;8%funding and5%borrow target","capacity_impact_separate":True,"comfortable_capacity":None,"soft_capacity":None,"strained_capacity":None,"hard_capacity":None,"unresolved_reason":"No auction participation or borrow availability/capacity measurement;1mreference is not cleared AUM"},
        "limitations":spec_dict["known_limits"],"next_tests":["Fixed physical-order combined-book replay at client capital and common price vintage","Liquidity/borrow/cost capacity test","Prospective locked monitoring without retrospective retuning"],
        "sources":["source_manifest.json","source_audit_addendum.json","catalog_complete.json"],
        "adaptive_lineage":{"profile":"standard","rounds_completed":1,"declared_total_variants":318,"actual_total_variants":318,"active_minutes_used":state_dict["runtime_budget"]["active_minutes_used"],"stop_reason":"Original300 plus18 post-result sector comparison cells complete; no historical holdout; retain original forward hypotheses without further tuning"},
        "artifacts":{"concise_report":"REPORT.md","html_report":"REPORT.html","full_report":"REPORT_FULL.md","notebook":"decision_notebook.ipynb","frozen_specification":"research_spec_effective.json","manifest":"run_manifest.json","primary_source_code":["../../../scripts/research/portfolio_family_20260923/analyze.py","../../../scripts/research/portfolio_family_20260923/core_replay.py"],"primary_tables":["tables/portfolio_metrics.csv","tables/frozen_gate_results.csv"],"primary_charts":["charts/wealth.png","charts/drawdown.png"],"research_state":"research_state.json","hypothesis_registry":"hypothesis_registry.json","experiment_ledger":"experiment_ledger.jsonl","decision_log":"decision_log.jsonl","source_rule_map":"SOURCE_RULE_MAP.md"},
        "tags":["research_only","USD","no_live_changes","portfolio","CORE5","seen_history"]}
    (STUDY_PATH/"knowledge_record.json").write_text(json.dumps(record_dict,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"appendix":"REPORT_FULL.html","eligible_count":len(catalog_dict),"knowledge":"knowledge_record.json"}))


if __name__=="__main__":
    main()
