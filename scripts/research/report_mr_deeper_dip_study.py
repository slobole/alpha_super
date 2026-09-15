"""Build Hebrew research reports and a reproducible saved-artifact notebook."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pandas as pd
REPO_PATH=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_PATH))
from scripts.research.run_mr_deeper_dip_study import STUDY_PATH, STRATEGY_TUPLE, sha256_file, write_json
TABLE_PATH=STUDY_PATH/"tables"
NAME_DICT={"dv2":"DV2","hpi235":"HPI בהצבעת 2/3/5","sector":"סקטורים VOX/IYR"}
POLICY_DICT={"moo":"פתיחה","limit_0.5pct":"לימיט ‎0.5%","limit_1pct":"לימיט ‎1%","limit_2pct":"לימיט ‎2%"}

def table(frame_df):
    header_list=[str(value_obj) for value_obj in frame_df.columns]
    row_list=["| "+" | ".join(header_list)+" |", "| "+" | ".join(["---"]*len(header_list))+" |"]
    for value_tuple in frame_df.itertuples(index=False,name=None):
        row_list.append("| "+" | ".join(str(value_obj).replace("|","/").replace(chr(10)," ") for value_obj in value_tuple)+" |")
    return chr(10).join(row_list)

def formatted(frame_df, column_dict, percent_list=(), number_list=()):
    output_df=frame_df[list(column_dict)].copy()
    for column_str in percent_list:
        output_df[column_str]=output_df[column_str].map(lambda value_float:f"{value_float*100:.2f}%")
    for column_str in number_list:
        output_df[column_str]=output_df[column_str].map(lambda value_float:f"{value_float:,.2f}")
    if "strategy" in output_df:
        output_df["strategy"]=output_df["strategy"].map(NAME_DICT)
    if "policy" in output_df:
        output_df["policy"]=output_df["policy"].map(POLICY_DICT)
    return table(output_df.rename(columns=column_dict))

def build_reports():
    summary_df=pd.read_csv(TABLE_PATH/"summary.csv")
    expected_set={(strategy_str,layer_str,policy_str) for strategy_str in STRATEGY_TUPLE
                  for layer_str in ("central","stress") for policy_str in POLICY_DICT}
    actual_list=list(summary_df[["strategy","layer","policy"]].itertuples(index=False,name=None))
    if len(actual_list)!=24 or set(actual_list)!=expected_set:
        raise AssertionError("Report requires exactly the24 expected cells.")
    period_df=pd.read_csv(TABLE_PATH/"subperiods.csv")
    paired_df=pd.read_csv(TABLE_PATH/"paired_summary.csv")
    inference_df=pd.read_csv(TABLE_PATH/"inference.csv")
    capacity_df=pd.read_csv(TABLE_PATH/"capacity_summary.csv")
    padding_df=pd.read_csv(TABLE_PATH/"dv2_input_exposure_summary.csv")
    timing_df=pd.read_csv(TABLE_PATH/"timing_attribution.csv")
    central_df=summary_df.loc[summary_df["layer"].eq("central")].copy()
    stress_df=summary_df.loc[summary_df["layer"].eq("stress")]
    baseline_df=central_df.loc[central_df["policy"].eq("moo")].set_index("strategy")
    conclusion_list=[]
    delta_list=[]
    for strategy_str in STRATEGY_TUPLE:
        group_df=central_df.loc[central_df["strategy"].eq(strategy_str)]
        limit_df=group_df.loc[~group_df["policy"].eq("moo")]
        for row_obj in limit_df.itertuples(index=False):
            stress_row=stress_df.loc[(stress_df["strategy"]==strategy_str)&(stress_df["policy"]==row_obj.policy)].iloc[0]
            stress_base=stress_df.loc[(stress_df["strategy"]==strategy_str)&(stress_df["policy"]=="moo")].iloc[0]
            group_period_df=period_df.loc[(period_df["strategy"]==strategy_str)&(period_df["layer"]=="central")]
            base_period_ser=group_period_df.loc[group_period_df["policy"]=="moo"].set_index("period")["cagr"]
            arm_period_ser=group_period_df.loc[group_period_df["policy"]==row_obj.policy].set_index("period")["cagr"]
            delta_list.append({"strategy":strategy_str,"policy":row_obj.policy,
                "central_delta":row_obj.cagr-baseline_df.loc[strategy_str,"cagr"],
                "stress_delta":stress_row["cagr"]-stress_base["cagr"],
                "positive_periods":int((arm_period_ser>base_period_ser).sum())})
    delta_df=pd.DataFrame(delta_list)
    delta_df.to_csv(TABLE_PATH/"decision_summary.csv",index=False)
    any_cagr_improvement_bool=bool(delta_df["central_delta"].gt(0.).any())
    # This report describes the frozen family; these are not promotion gates.
    dv2_limit_ser=central_df.loc[(central_df["strategy"]=="dv2")&(central_df["policy"]=="limit_1pct")].iloc[0]
    hpi_limit_ser=central_df.loc[(central_df["strategy"]=="hpi235")&(central_df["policy"]=="limit_0.5pct")].iloc[0]
    conclusion_list=[
        f"**DV2:** לימיט 1% כמעט שומר על התשואה השנתית ({dv2_limit_ser['cagr']:.2%} מול {baseline_df.loc['dv2','cagr']:.2%}), ומשפר את השארפ ואת הירידה המרבית. זהו שינוי ביחס בין סיכון לתשואה, עם תלות חזקה בתקופה ובעלות.",
        f"**HPI בהצבעת 2/3/5:** לימיט 0.5% מעלה את התשואה השנתית מ־{baseline_df.loc['hpi235','cagr']:.2%} ל־{hpi_limit_ser['cagr']:.2%}. עיקר היתרון נוצר בשנים 2015–2019 ותלוי גם בשינוי מסלול העסקאות של התיק.",
        "**סקטורים VOX/IYR:** כל שלושת העומקים מורידים את התשואה ואת השארפ. הירידה בחשיפה אינה מפצה על העסקאות שהוחמצו; התוצאה שלילית בכל ארבע תקופות המשנה ובשתי שכבות העלות."
    ]
    verdict_str="אין שיפור אחיד מעצם ההמתנה למחיר נמוך יותר. HPI בלימיט 0.5% מציג שיפור היסטורי, ו־DV2 בלימיט 1% מציג פשרה מעניינת בין תשואה לסיכון; בסקטורים הניסוי אינו תומך בהחלפת הכניסה בפתיחה. אף חלופה אינה מאומתת לאימוץ תפעולי."
    central_table_str=formatted(central_df,{
        "strategy":"מערכת","policy":"כניסה","cagr":"תשואה שנתית","sharpe":"שארפ",
        "max_drawdown":"ירידה מרבית","fill_rate":"שיעור מילוי","mean_exposure":"חשיפה ממוצעת"},
        ("cagr","max_drawdown","fill_rate","mean_exposure"),("sharpe",))
    delta_table_str=formatted(delta_df,{
        "strategy":"מערכת","policy":"לימיט","central_delta":"פער שנתי בסיסי",
        "stress_delta":"פער שנתי בלחץ","positive_periods":"תקופות עם יתרון מתוך 4"},
        ("central_delta","stress_delta"))
    paired_table_str=formatted(paired_df,{
        "strategy":"מערכת","policy":"לימיט","fill_rate":"מילוי המקור",
        "mean_baseline_return_all_opportunities":"ממוצע להזדמנות בפתיחה",
        "mean_limit_return_all_opportunities":"ממוצע להזדמנות בלימיט",
        "delta_pnl":"שינוי רווח מזווג $"},("fill_rate","mean_baseline_return_all_opportunities","mean_limit_return_all_opportunities"),
        ("delta_pnl",))
    market_table_str=formatted(central_df,{
        "strategy":"מערכת","policy":"כניסה","daily_correlation":"מתאם יומי",
        "monthly_correlation":"מתאם חודשי","beta":"בטא"},(),("daily_correlation","monthly_correlation","beta"))
    headline_str="\n\n".join(conclusion_list)
    state_dict=json.loads((STUDY_PATH/"research_state.json").read_text(encoding="utf-8"))
    runtime_minutes_int=int((datetime.now(timezone.utc)-datetime.fromisoformat(state_dict["created_at"])).total_seconds()/60.)
    report_str=rf"""# האם כדאי לחכות לירידה עמוקה יותר לפני כניסה?

## סיכום

{verdict_str}

נבדקו **24 חלופות שנקבעו מראש**: שלוש מערכות חזרה לממוצע, ארבע מדיניות כניסה ושתי שכבות עלות. החלון המשותף הוא **5 בינואר 2010 עד 14 בספטמבר 2026**, ובו 4,198 ימי מסחר. כל תיק התחיל ב־100,000 דולר והתפתח בנפרד. זהו ניסוי ברכיב הכניסה של המערכות הקיימות; האיתותים, הדירוג, כללי חישוב כמות המניות וכללי היציאה נשמרו. הכמויות בפועל משתנות עם שוויו של כל תיק.

{headline_str}

השאלה החשובה היא האם מחיר זול יותר מפצה על עסקאות שלא מתמלאות ועל הזמן שבו ההון נשאר במזומן. תשואה ממוצעת טובה יותר בעסקאות שהתמלאו, כשלעצמה, אינה תשובה לשאלה הזאת.

## מה לקחנו מהמאמר

המאמר *Where to Buy the (Deeper) Dip* של Concretum Research מציג כניסות מתחת לפתיחת היום הבא לאחר שפל של עשרה ימים. ככל שהלימיט עמוק יותר, מספר העסקאות קטן והתשואה הממוצעת לעסקה המדווחת עולה. לדוגמה, בטבלה שבמאמר כניסה בפתיחה כוללת 71,091 עסקאות עם ממוצע של 29 נקודות בסיס; ב־1% מתחת לפתיחה יש 39,801 עסקאות וממוצע של 46 נקודות בסיס.

כאן נבדקה העברת **רעיון הכניסה** לשלוש המערכות שביקשת. אסטרטגיית שפל עשרת הימים של המאמר לא שוחזרה. עומק 0.5% הוא תוספת שביקשת; לא נוספו עומקים אחרי צפייה בתוצאות. השוואת הבסיס לקוד הקיים עברה התאמה מלאה, אבל אין בכך אישור לכל הנחות הנתונים של הקוד, ובפרט למגבלת החברות במדד של DV2.

## מה בדיוק השתנה

איתות ותקציב נקבעים בסגירת יום T. בגרסת הבסיס מתבצעת קנייה בפתיחת T+1. בגרסת הלימיט רואים תחילה את הפתיחה ומציבים מחיר נמוך ממנה באחוז הקבוע. לדוגמה, לפתיחה של 100 דולר ועומק 1% מתאים לימיט של 99 דולר. הוא בתוקף לאותו יום בלבד.

$$
L_{{T+1}}=O_{{T+1}}(1-d),\qquad d\in\{{0.005,0.01,0.02\}}
$$

$$
\mathrm{{Fill}}_{{T+1}}=\mathbf{{1}}\{{Low_{{T+1}}\le L_{{T+1}}\}},\qquad P_{{buy}}=L_{{T+1}}
$$

אם אין מילוי, הכסף נשאר במזומן. אין קנייה מאוחרת במחיר שוק ואין החלפה תוך־יומית במניה שהגיעה ללימיט. בסגירה הבאה המערכת בוחנת מחדש את האיתות המקורי. כך החמצת עסקה יכולה לשנות גם את בחירת העסקאות בעתיד; לכן לכל חלופה נוהל תיק עצמאי. בכל החלופות איתות היציאה הרגילה מתקבל בסגירה והמכירה מתבצעת בפתיחת היום הבא; סגירה מנהלית בהיעדר מחיר מתועדת בנפרד.

```text
[סגירת T: איתות, דירוג וכמות]
                  |
       *** CRITICAL *** מידע זמין עד T
                  v
[פתיחת T+1: קביעת מחיר הלימיט]
                  |
          [בדיקת מילוי ביום]
             /           \
          התמלא          פקע
            |              |
      [פוזיציה]         [מזומן]
            |              |
      [יציאה מקורית] [בחינה בסגירה הבאה]
```

**מגבלה מהותית:** המחיר בפתיחה ידוע רק לאחר הפתיחה. נר יומי אינו אומר אם השפל הגיע לפני שהפקודה יכלה להגיע לבורסה. המודל מניח הפעלה מיידית אחרי הפתיחה ומילוי מלא. הוא אינו מוכיח מילוי אמיתי, מקום בתור או היעדר מילוי חלקי.

## התוצאה בתיק

הטבלה כוללת עלויות, דיבידנדים נטו ומזומן. שארפ מחושב מכל ימי המסחר ובריבית חסרת סיכון אפס; ירידה מרבית נמדדת משיא קודם של שווי התיק. שיעור המילוי הוא מתוך הוראות הכניסה של כל תיק, ולכן המכנה משתנה בין החלופות.

{central_table_str}

![עקומות הון וירידה מהשיא](charts/equity_drawdown.png)

עקומת המדד היא S&P 500 בתשואה כוללת, ללא העמלות וניכוי המס שמוחלים בסימולציית המערכות. היא מוצגת להקשר; המערכות אינן מחזיקות חשיפה מלאה וקבועה למדד. הפחתה בירידה המרבית שמגיעה יחד עם ירידה גדולה בחשיפה אינה מספיקה לבדה כדי לקרוא לשינוי שיפור. לכן מוצגים גם שארפ, ניצול ההון והתשואה.

![שיעור מילוי וחשיפה](charts/fills_exposure.png)

## האם היתרון מחזיק בתקופות ובעלויות?

לפני ההרצות הוגדרו ארבע תקופות: 2010–2014, 2015–2019, 2020–2022 ו־2023 עד סוף המדגם. בטבלה הבאה הפער הוא תשואת הלימיט פחות תשואת הכניסה בפתיחה, בנקודות אחוז לשנה. מספר תקופות חיוביות מסכם את כיוון ההשוואה, ואינו מבחן אימות עצמאי.

{delta_table_str}

ב־HPI בלימיט 0.5%, היתרון השנתי על הבסיס הוא כ־0.69 נקודת אחוז בשנים 2010–2014 וכ־7.19 בשנים 2015–2019, אך כ־1.43− בשנים 2020–2022 ורק כ־0.03 בשנים 2023–2026. לכן ספירת שלוש תקופות חיוביות מסתירה ריכוז משמעותי בתקופה אחת. היתרון הכולל של 0.5% נשמר גם בתרחיש הלחץ: כ־11.57% לשנה מול 9.62% בבסיס, עם שארפ טוב יותר; גם שם 2015–2019 מספקת את עיקר היתרון. בשכבת העלות הבסיסית, לימיט 2% מפחית את תשואת HPI בכל ארבע התקופות. בתרחיש הלחץ שלו התשואה הכוללת עדיין נמוכה מהבסיס, 9.28% מול 9.62%, אך השארפ והירידה המרבית טובים יותר ושלוש תקופות מציגות פער תשואה חיובי. לימיט 1% של HPI משפר תשואה ושארפ בשתי שכבות העלות, אך בתרחיש הלחץ הירידה המרבית שלו גרועה מעט מהבסיס.

גם DV2 בלימיט 1% אינו יציב בין התקופות: הפערים הם כ־4.68−, 1.12+, 4.92− ו־8.71+ נקודות אחוז, בהתאמה. בתרחיש הלחץ התשואה השנתית שלו היא כ־8.72% מול 8.37% לבסיס, ובשתי שכבות העלות השארפ והירידה המרבית טובים יותר. ההפחתה במסחר חשובה כאשר העלויות גבוהות; אין מכאן אומדן של עלויות המסחר האמיתיות.

בתרחיש הבסיס עלות המחיר היא 2.5 נקודות בסיס לכל צד, בתוספת 0.005 דולר למניה ומינימום דולר לפקודה. במילוי לימיט המחיר נשאר בדיוק במחיר המגביל; עלות מחקר מקבילה של 2.5 נקודות בסיס נגבית בנפרד מהמזומן. כך מחיר העסקה אינו חורג מהלימיט. עיגול המניות והעמלה למניה משתמשים ביחידות המודל המתואמות לאירועי הון, כפי שעושה מנוע הבסיס; לא שוחזר מספר המניות המקורי בכל תאריך. לכן זו עלות מחקר, ולא שחזור של עמלות ברוקר היסטוריות.

בתרחיש הלחץ נוספו 10 נקודות בסיס לכל צד שבוצע, ונדרש שהשפל יעבור עוד 5 נקודות בסיס מתחת ללימיט. אי־מילוי אינו מחויב בעמלה. זו בדיקת רגישות משותפת למחיר ולמילוי, והיא אינה אומדן מכויל של הבורסה או חסם מתמטי לתשואה.

![רגישות לעלות ולמילוי](charts/cost_sensitivity.png)

## המחיר המשופר מול העסקאות שהוחמצו

נוסף לתיקים העצמאיים נעשתה השוואה מזווגת: אותן כניסות שבוצעו בבסיס, אותן כמויות ואותן יציאות. חמש הוראות הבסיס של HPI שבוטלו בגלל היעדר פתיחה אינן נכללות בבדיקה המזווגת; הן כן נכללות בסימולציית התיק. עסקה שאינה מגיעה ללימיט מקבלת רווח אפס. כך אפשר להפריד בין חיסכון במחיר לבין ויתור על עסקאות, בלי לייחס את השינוי לריבית דריבית או לרשימת מניות אחרת.

{paired_table_str}

ממוצע התשואה נותן לכל הזדמנות מקורית משקל שווה ומשתמש בשווי הכניסה המקורי בפתיחה כמכנה משותף. הסכומים הדולריים משתמשים בכמויות תיק הבסיס כפי שהתפתחו במשך השנים, ולכן נותנים יותר משקל לתקופות שבהן פוזיציות הבסיס גדולות יותר. הם כוללים גם את הפרש עלות הכניסה. הם אבחון של הזדמנויות קבועות, ואינם רווח של תיק חדש שניתן לחבר לעקומת ההון. עסקאות שעדיין פתוחות בסיום מסומנות לפי שווי השוק ומופרדות מעסקאות סגורות בקבצים.

ההבחנה הזאת משנה את פירוש התוצאה: ב־HPI בלימיט 0.5% התיק העצמאי השתפר, אך על אותה רשימת עסקאות מקורית הרווח קטן בכ־160 אלף דולר. מתוך 4,075 כניסות הלימיט בפועל, 884 הן בצירופי מניה ותאריך שאינם מופיעים בבסיס. כלומר, השיפור תלוי גם בהמשך מסלול התיק, ולא מוסבר רק במחיר קנייה נמוך יותר. זו אבחנה, ואינה בידוד סיבתי של תרומת המקומות שהתפנו לעומת שינוי בכמויות.

בכל תשע ההשוואות המזווגות ממוצע התשואה להזדמנות מקורית ירד. ב־DV2 בלימיט 0.5% הסכום הדולרי דווקא עלה בכ־35 אלף דולר, בגלל משקלן השונה של העסקאות לאורך השנים; בתיק העצמאי התשואה השנתית ירדה. שתי הבדיקות נדרשות כדי לא לבלבל בין מחיר טוב יותר, בחירה שונה וניצול הון.

## מה נשמר בכל מערכת

- **DV2:** מדד DV2 בחלון 126 נמוך מ־10, סגירה מעל ממוצע 200 יום ותשואת 126 ימים מעל 5%. הדירוג הוא תנודתיות מנורמלת ל־14 ימים בסדר יורד. עד עשר פוזיציות; יציאה כאשר הסגירה גבוהה מהגבוה של היום הקודם.
- **HPI בהצבעת 2/3/5:** לפחות שניים משלושת האופקים מציגים תשואה שלילית ו־HPI נמוך מ־30. נוסף לכך, מיקום הסגירה בטווח היום נמוך מ־0.10 והמחיר מעל ממוצע 200. הדירוג לפי מחזור כספי. עד עשר פוזיציות; יציאה במיקום סגירה מעל 0.90, RSI של יומיים מעל 90, או אובדן חברות במדד.
- **סקטורים:** מיקום הסגירה בטווח קטן מ־0.05, והירידה מהסגירה הקודמת גדולה מחצי ATR של היום הקודם. הדירוג לפי תנודתיות מנורמלת קודמת. יציאה דורשת מיקום סגירה מעל 0.90 וטווח לוגריתמי מעל חציון 21 הימים הקודמים. הסל כולל את תשע קרנות הסקטורים הוותיקות וכן VOX ו־IYR.

במניות הכמות היא החלק השלם של עשירית שווי התיק בסגירה חלקי מחיר הסגירה. בסקטורים נשמרו מניות חלקיות והקצאה של 1.5/11 לכל פוזיציה, עד חמש פוזיציות. לפיכך החשיפה המרבית המתוכננת היא כ־68.18%, ולא 150%. לא בוצע איזון שוטף של פוזיציות קיימות.

## מה אפשר להסיק — ומה עדיין חסר

המסקנה היא על ההשוואה ההיסטורית המסוימת הזאת. לא נשמרה כאן תקופת מבחן שלא נצפתה בעבר, ואין הצהרה על תשואה עתידית. בוצעו תשע השוואות עיקריות ותיקון Holm למובהקות המשפחתית; רווחי הסמך הם לתוחלת הפרש התשואה היומית, ולא להפרש התשואה השנתית המצטברת. כל המשפחה מדווחת, בלי לבחור רק את העומק המוצלח. בפועל, לא נמצא שיפור חיובי מובהק בתוחלת התשואה היומית באף חלופה מנייתית לאחר התיקון. ב־HPI בלימיט 0.5% ערך p המתוקן הוא כ־0.88; רווח הסמך הלא־מתוקן כולל אפס. הפגיעה בתוחלת התשואה של שלושת לימיטי הסקטורים כן נשארת מובהקת לאחר התיקון. מבחנים אלה אינם מבחני מובהקות לשארפ, לירידה המרבית או ל־CAGR.

ב־DV2 נשמר מנגנון הבסיס שמסיר בדיעבד את חמשת ימי החברות האחרונים של חברות שכבר אינן במדד: 1,610 ימי־מניה בתקופת הניסוי. נשמרו גם מוסכמות השלמת המחירים המקוריות. טבלאות הביקורת מפרטות את ההחלטות והביצועים שנחשפו לתצפיות כאלה; היעדר דגל בחלון 200 ימים אינו שולל השפעה רחוקה יותר של ATR רקורסיבי.

תזרים הדיבידנדים נשמר לפי מניות שהוחזקו בסגירה הקודמת, בניכוי 25%. קונה ביום האקס אינו מקבל את החלוקה, ומוכר באותו יום שומר עליה. מזומן מקבל תשואה אפס, ועלות מימון יתרות שליליות אינה ממודלת. לכן מוצגים מספר ימי המזומן השלילי והמינימום בדוח המלא.

בדיקת ההשתתפות במחזור משתמשת במחזור כספי ממוצע של 63 ימים שהיה ידוע בסגירה. היא בודקת את דרישת הפקודות שנבחרו, כולל אלה שלא התמלאו, בהון ההתחלתי שנקבע ובהתפתחותו. היא אינה אישור לקיבולת, לביצוע חלקי או לגודל תיק אחר.

**הכרעה:** {verdict_str} בדיקת המשך של חלופה קיימת בעלת יתרון היסטורי צריכה לבחון זמן הפעלה אמיתי בנתונים תוך־יומיים ובהמשך מדגם עצמאי שלא נצפה. רעיון כניסה חדש דורש הגדרה נפרדת מראש. סריקת עוד אחוזים על אותה היסטוריה לא תהפוך אותה לאימות עצמאי.

## קבצים ואפשרות שחזור

[דוח מלא](REPORT_FULL.md) · [מחברת עם פלטים](decision_notebook.ipynb) · [כל 24 התוצאות](tables/summary.csv) · [מפרט קפוא](research_spec_frozen.json) · [ייצוג סכמה נגזר](research_spec_normalized_view.json) · [מקור הכללים](SOURCE_RULE_MAP.md) · [ביקורת נתונים](DATA_AUDIT.md) · [חתימות הקבצים](run_manifest.json)

### רישום ביקורת לשחזור

המפתחות הבאים משמשים את כלי הבדיקה; זמן הריצה כולל גם המתנה, ולכן הוא חסם עליון שמרני לזמן עבודה פעיל. המפרט המקורי נשאר ללא שינוי; ייצוג הסכמה הנגזר הוא תרגום מאוחר של שדות התיעוד בלבד.

```yaml
scope: research-only
verdict: diagnostic
replication: שלושת בסיסי הקוד בלבד
validation: היסטוריה מוכרת ללא מדגם אימות עצמאי
runtime: {runtime_minutes_int} דקות שחלפו; תקרה קשיחה 180 דקות
source: SOURCE_RULE_MAP.md
timing: Close_T -> Open_T+1 או לימיט יומי מותנה
search: 24 תאים קבועים מראש; אפס סבבי התאמה
holdout: לא הוקצה מדגם שלא נצפה
failure: מגבלות הנתונים והמילוי מפורטות ב־DATA_AUDIT.md
artifact: run_manifest.json
```

הקוד והבדיקות נשמרו במחקר המבודד. לא השתנו כללי אסטרטגיות משותפים או נתיבי מסחר. הבדיקה העצמאית כיסתה התאמה לבסיס, מילויים, כמויות, עמלות, דיבידנדים, מזומן, שווי תיק והפרדה בין עסקאות סגורות להחזקות בסוף המדגם.
"""
    (STUDY_PATH/"REPORT.md").write_text(report_str,encoding="utf-8")
    full_str=report_str+rf"""

# נספחים מלאים

## כל תרחישי הלחץ

{formatted(stress_df,{"strategy":"מערכת","policy":"כניסה","cagr":"תשואה שנתית","sharpe":"שארפ","max_drawdown":"ירידה מרבית","fills":"מילויים","orders":"פקודות"},("cagr","max_drawdown"),("sharpe",))}

## עסקאות, זנבות ומימון

{formatted(central_df,{"strategy":"מערכת","policy":"כניסה","closed_trades":"סגורות","open_trades":"פתוחות בסוף","mean_trade_return":"ממוצע עסקה","mean_loss":"ממוצע הפסד","trade_cvar5":"ממוצע 5% הגרועות","mean_duration":"ימי החזקה","negative_cash_days":"ימי מזומן שלילי","min_cash_weight":"מזומן מזערי"},("mean_trade_return","mean_loss","trade_cvar5","min_cash_weight"),("mean_duration",))}

ממוצעי העסקאות והזנבות בטבלה מתייחסים לעסקאות סגורות בלבד. כל השוואות התיק כוללות את שווי ההחזקות שנותרו בסיום; רווחן והסכום המסומן מופיעים ב־summary.csv. אין מכירה מדומה בסוף המדגם.

## סיכון, מחזור ומזומן

{formatted(central_df,{"strategy":"מערכת","policy":"כניסה","volatility":"תנודתיות שנתית","annual_turnover":"מחזור שנתי פי הון","final_nav":"שווי סופי $","native_forced_liquidations":"סגירות מנהליות","mean_borrowed_weight":"יתרה שלילית ממוצעת / הון"},("volatility","mean_borrowed_weight"),("annual_turnover","final_nav"))}

היתרה השלילית הממוצעת מחושבת על כל ימי המדגם, עם אפס בימים שאין יתרה שלילית. אין זו עמלת מימון שנגבתה בפועל; כל מודל העלויות בדוח גלוי במפרט.

## מתאם לשוק ובטא

{market_table_str}

מתאם יומי וחודשי משתמשים באותו מדד כולל דיבידנדים ובאותם תאריכים. בטא היא יחס השונות המשותפת של תשואות המערכת והמדד לשונות תשואות המדד.

![מתאם מתגלגל ל־126 ימי מסחר](charts/rolling_correlation.png)

## כל תקופות המשנה

{formatted(period_df,{"strategy":"מערכת","layer":"עלות","policy":"כניסה","period":"תקופה","cagr":"תשואה שנתית","sharpe":"שארפ","max_drawdown":"ירידה מרבית"},("cagr","max_drawdown"),("sharpe",))}

## פירוק דולרי של ההזדמנויות המקוריות

{formatted(paired_df,{"strategy":"מערכת","policy":"לימיט","net_fill_price_improvement":"שיפור מחיר נטו $","missed_baseline_pnl":"רווח בסיס שהוחמץ $","delta_pnl":"שינוי נטו $"},(),("net_fill_price_improvement","missed_baseline_pnl","delta_pnl"))}

השינוי הוא שיפור מחיר הכניסה נטו פחות הרווח המקורי שהוחמץ. אם הרווח שהוחמץ שלילי, ההימנעות היא תרומה חיובית. ממוצע שווה־משקל וסכום דולרי יכולים להצביע לכיוונים שונים עקב גדלי הפוזיציות המשתנים.

## שינוי במסלול בחירת העסקאות

{table(pd.read_csv(TABLE_PATH/"entry_path_comparison.csv").round(6))}

התאמה פירושה אותו נכס ואותו יום כניסה. תדירות מילוי מכסת הפוזיציות נמדדת בסוף יום, ולא בשיא תוך־יומי. זו אבחנה תיאורית: היא אינה מפרידה סיבתית בין מקומות שהתפנו, שינוי דירוג בפועל ושינוי בכמויות עקב שווי תיק שונה.

## השוואות סטטיסטיות

{table(inference_df.round(6))}

שגיאות התקן מתוקנות בשיטת HAC עם חמש השהיות. ערכי p דו־צדדיים; העמודה p_holm_9 מתקנת תשע השוואות. רווחי הסמך של 95% לפני תיקון משפחתי, ותוחלת ההפרש מוכפלת ב־252 לצורכי הצגה. אין אלה רווחי סמך ל־CAGR. חשיפה שונה ומשטרי שוק משותפים נשארים חלק מהשאלה הכלכלית.

## פירוק עיתוי

{table(timing_df.round(6))}

זהו ממוצע לפי הזדמנויות הבסיס, כולל הסימון בסוף המדגם. מחיר היציאה כולל את החיכוך המקורי. הדיבידנדים והעמלות אינם בפירוק המחירים הזה. הזהות נבדקת לכל עסקה: תשואת סגירה־פתיחה והתשואה מפתיחה ליציאה מתחברות בכפל, לא בחיבור פשוט. אף זרוע אינה מקבלת את רווח הלילה המסוים שבין איתות הכניסה לפתיחת יום הכניסה; לילות מאוחרים יותר במהלך ההחזקה כלולים כרגיל.

## ביקורת השלמת נתונים ב־DV2

{table(padding_df)}

הסימון משווה לתצפיות OHLC מלאות במקור HPI ללא השלמת ימים. מניה שאינה קיימת במקור ההשוואה מסומנת כלא ידועה, ולא כתקינה או מלאכותית. פירוט ברמת הוראה וביצוע: dv2_input_exposure_orders.csv ו־dv2_input_exposure_transactions.csv. מכירה מנהלית בהיעדר מחיר מזוהה בנפרד ממכירה רגילה בפתיחה.

## השתתפות במחזור

{table(capacity_df.round(7))}

המספרים הם יחסים, ולכן 0.01 משמעו 1% מהמחזור היומי הממוצע. דרישת הכניסה המצטברת לאותה מניה באותו יום נשמרה גם על פני המערכות. כשיש הבדל בין אומדני מחזור ממקורות הטעינה, נעשה שימוש במכנה הקטן יותר והפער מסומן. מחזור יומי אינו המחזור הזמין בזמן נגיעה בלימיט. לא בוצעה סריקת הון ולא ניתן סיווג קיבולת תפעולי.

## נוסחאות המדדים ושחזור

$$
r_t=E_t/E_{{t-1}}-1,\quad E_0=100000,\quad
CAGR=(E_N/E_0)^{{252/N}}-1
$$

$$
Sharpe=\sqrt{{252}}\,\bar r/s_r,\quad
DD_t=E_t/\max(E_0,\ldots,E_t)-1
$$

שונות המדגם משתמשת ב־N−1. מחזור המסחר הוא סכום ערך הביצועים המוחלט חלקי שווי תיק ממוצע, מוכפל ב־252/N; הוא כולל קניות ומכירות. אין תשואה חסרה שנמחקת בתחילת העקומה.

### פקודות שחזור

```powershell
# Use the frozen local inputs; this does not fetch a new vendor snapshot.
.venv/Scripts/python.exe -B scripts/research/run_mr_deeper_dip_study.py run sector
.venv/Scripts/python.exe -B scripts/research/run_mr_deeper_dip_fast.py dv2
.venv/Scripts/python.exe -B scripts/research/run_mr_deeper_dip_fast.py hpi235
.venv/Scripts/python.exe -B scripts/research/analyze_mr_deeper_dip_study.py
.venv/Scripts/python.exe -B scripts/research/report_mr_deeper_dip_study.py
.venv/Scripts/python.exe -B -m pytest tests/test_mr_deeper_dip_study.py -q
```

הפעלה חוזרת מאמתת חתימות ואינה מוסיפה חלופה כלכלית. מאיץ ההרצה משנה רק העתקת מיפוי מטא־נתונים בזיכרון. לפני הרצות המניות הוכחה התאמה מלאה להרצת הבסיס המקורית על כל התקופה. אסמכתאות המאיץ מקשרות את קוד המעטפת לתאים שהורצו בה.

## גבולות הביקורת

אין טענה לשחזור מלא של המאמר, להיעדר כל הטיה בקוד הבסיס, למדגם שלא נראה בעבר או למילוי אמיתי. התאמת הכספים היא בדיקת חשבונאות. בדיקת קידומת התכונות מכסה 12 מניות בכל מערכת מנייתית ושתי נקודות זמן, ואת כל הקרנות בסקטורים; היא אינה הוכחה ממצה לכל אפשרות. ארבע תקופות המשנה קבועות מראש ואינן ארבע בדיקות בלתי תלויות.
"""
    (STUDY_PATH/"REPORT_FULL.md").write_text(full_str,encoding="utf-8")
    return verdict_str, any_cagr_improvement_bool, summary_df, delta_df



def write_normalized_spec_view():
    """Serialize the frozen contract to the skill schema; never change its bytes."""
    from copy import deepcopy
    import importlib.util
    original_path=STUDY_PATH/"research_spec_frozen.json"
    original_digest_str=sha256_file(original_path)
    original_dict=json.loads(original_path.read_text(encoding="utf-8"))
    view_dict=deepcopy(original_dict)
    view_dict["original_contract"]=deepcopy(original_dict)
    view_dict["serialization_metadata"]={
        "original_location":str(original_path),"original_content_id":"sha256:"+original_digest_str,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "initial_frozen_at_original":original_dict["initial_frozen_at"],
        "purpose":"Post-result schema serialization view only; no new freeze, threshold, variant, cost, timing or promotion decision."}
    view_dict["initial_frozen_at"]="2026-09-15T15:58:52+00:00"
    view_dict["sources"]=[{**source_dict,"source_id":"sha256:"+source_dict["sha256"],
                          "location":source_dict["path"]} for source_dict in original_dict["sources"]]
    artifact_list=[]
    start_set=set()
    for strategy_str in STRATEGY_TUPLE:
        input_path=STUDY_PATH/"data"/strategy_str
        manifest_dict=json.loads((input_path/"input_manifest.json").read_text(encoding="utf-8"))
        start_set.add(manifest_dict["start"])
        artifact_list.extend({"location":str(input_path/name_str),"content_id":"sha256:"+digest_str,
                              "as_of":manifest_dict["database_session"]}
                             for name_str,digest_str in manifest_dict["sha256"].items())
    assert len(start_set)==1 and len(artifact_list)==8
    effective_start_str=next(iter(start_set))
    universe_list=["DV2 native SP500 with retrospective5day former-member tail exclusion",
                   "HPI235 exact historical SP500","Fixed11 US sector ETFs includingVOX/IYR"]
    view_dict["data"].update(period={
        "history_start":original_dict["data"]["history_start"],
        "requested_start":original_dict["data"]["requested_execution_start"],
        "effective_start":effective_start_str,"end":original_dict["data"]["execution_end"]},
        as_of=original_dict["data"]["data_asof"],input_artifacts=artifact_list,
        universes=universe_list,benchmarks=["$SPXTR"])
    view_dict["timing"].update(entry=original_dict["timing"]["baseline_entry"]+"; "+original_dict["timing"]["limit_entry"],
                              terminal_value=original_dict["timing"]["terminal"])
    view_dict["signal"].update(formula="; ".join(name_str+": "+value_str for name_str,value_str in original_dict["signal"].items()),
                              thresholds=[*original_dict["signal"].values(),{"entry_depths":original_dict["search_space"]["depths"]}])
    view_dict["feature_roles"]=[{"feature":key_str,"roles":[value_str]} for key_str,value_str in original_dict["feature_roles"].items()]
    portfolio_dict=original_dict["portfolio"]
    view_dict["portfolio"].update(engine="source-native run_daily stateful engine",
        maximum_positions=portfolio_dict["slots"],ranking=portfolio_dict["rank_and_exit"],
        sizing=portfolio_dict["quantities"],ensemble=portfolio_dict["state"],
        cash={key_str:portfolio_dict[key_str] for key_str in ("cash_return","financing","cash_constraints")})
    view_dict["evaluation"].update(periods={"discovery":original_dict["evaluation"]["window"],
        "validation":original_dict["evaluation"]["holdout"],"confirmation":original_dict["evaluation"]["holdout"],
        "full":effective_start_str+"/"+original_dict["data"]["execution_end"]},
        inference_unit=original_dict["evaluation"]["inference"],benchmark="$SPXTR")
    cost_dict=original_dict["costs"]
    basis_str="Nominal sum of entry and exit friction rates only; separate per-share/minimum commission and dividend withholding excluded. Not realized all-in cost."
    central_bps_float=20000.*cost_dict["central"]["market_price_slippage_per_side"]
    stress_bps_float=central_bps_float+20000.*cost_dict["stress"]["additional_cash_friction_every_executed_side"]
    view_dict["costs"].update(
        paper_like={"round_trip_bps":0.,"included_components":["Gross paired price decomposition only"],
                    "execution_status":"Descriptive gross component only; no extra full portfolio or economic cell."},
        central_research={"round_trip_bps":central_bps_float,"round_trip_bps_basis":basis_str,
            "included_components":["Market slippage or equivalent limit cash friction","Separate commission"],
            "literal_parameters":cost_dict["central"]},
        conservative_survival={"round_trip_bps":stress_bps_float,"round_trip_bps_basis":basis_str,
            "included_components":["Central costs","Additional executed-side cash friction","5bp limit penetration eligibility"],
            "literal_parameters":{"central":cost_dict["central"],"stress":cost_dict["stress"]}},
        capacity_impact={"separate_from_base":True,"calibrated":False,"formula":cost_dict["capacity"]},
        components=["Market slippage","Limit cash friction","Per-share/minimum commission",
                    "Stress cash friction","Dividend withholding: accounting, not execution friction"])
    view_dict["search_space"].update(declared_families=list(STRATEGY_TUPLE),
        axes={"strategy":list(STRATEGY_TUPLE),"depth":[None,*original_dict["search_space"]["depths"]],"layer":["central","stress"]},
        total_declared_variants=original_dict["search_space"]["total_economic_cells"])
    view_dict["promotion_rule"]={"economic":original_dict["promotion_rule"],"statistical":original_dict["evaluation"]["inference"],"risk":original_dict["promotion_rule"]}
    view_dict["evidence_waivers"]=[{"layer":key_str,"reason":value_str,"promotion_effect":original_dict["promotion_rule"]}
                                  for key_str,value_str in original_dict["evidence_waivers"].items()]
    view_dict["outputs"]={"concise_report":"REPORT.md","full_report":"REPORT_FULL.md","notebook":"decision_notebook.ipynb",
        "knowledge_record":"knowledge_record.json","manifest":"run_manifest.json","tables":"tables","charts":"charts"}
    view_path=STUDY_PATH/"research_spec_normalized_view.json"
    write_json(view_path,view_dict)
    validator_path=Path("C:/Users/User/.codex/skills/research-quant-signal-features/scripts/validate_research_spec.py")
    validator_spec=importlib.util.spec_from_file_location("spec_validator",validator_path)
    validator_obj=importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator_obj)
    failure_list=validator_obj.validate_research_spec(view_path)
    if failure_list:
        raise AssertionError(f"Normalized specification schema errors: {failure_list}")
    assert sha256_file(original_path)==original_digest_str
    print("PASS serialization-only normalized specification; original unchanged")


def write_notebook():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from tempfile import TemporaryDirectory
    notebook_obj=nbformat.v4.new_notebook()
    notebook_obj.metadata["kernelspec"]={"display_name":"Alpha research Python","language":"python","name":"research"}
    notebook_obj.cells=[
        nbformat.v4.new_markdown_cell("# ניסוי כניסה עמוקה יותר — מחברת החלטה\n\n24 חלופות קפואות; נתונים יומיים; אין תקופת אימות שלא נצפתה. המחברת קוראת תוצאות שמורות ואינה מריצה וריאנטים חדשים."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            +"candidate_list=[Path.cwd(), *Path.cwd().parents]\n"
            +"study_candidate_list=[path_obj/suffix_str for path_obj in candidate_list for suffix_str in ('', 'results/research/mr_deeper_dip_entry_study')]\n"
            +"study_path=next((path_obj for path_obj in study_candidate_list if (path_obj/'tables/summary.csv').is_file()), None)\n"
            +"assert study_path is not None, 'Open this notebook in the study folder or its repository, with the tables and charts folders present.'\n"
            +"summary_df=pd.read_csv(study_path/'tables/summary.csv')\n"
            +"assert len(summary_df)==24\nassert not summary_df.duplicated(['strategy','layer','policy']).any()\n"
            +"display(summary_df[['strategy','layer','policy','cagr','sharpe','max_drawdown','fills','orders','mean_exposure']])"),
        nbformat.v4.new_markdown_cell("## מחיר הכניסה מול ההזדמנויות שהוחמצו\n\nהבדיקה המזווגת מחזיקה את כמויות הבסיס ואת היציאות קבועות. היא אבחון, ולא תיק נוסף."),
        nbformat.v4.new_code_cell("display(pd.read_csv(study_path/'tables/paired_summary.csv'))\ndisplay(pd.read_csv(study_path/'tables/decision_summary.csv'))"),
        nbformat.v4.new_markdown_cell("## ביקורת, תקופות ועלויות\n\nכל החשבונות נבדקים מתוך קובצי העסקאות. תיקון המובהקות מתייחס לתשע השוואות יומיות; אין אימות מחוץ למדגם."),
        nbformat.v4.new_code_cell(
            "assert summary_df['cash_error_max'].max()<1e-5\nassert summary_df['trade_nav_error'].max()<1e-5\n"
            +"assert summary_df['daily_mark_error'].max()<1e-5\n"
            +"display(pd.read_csv(study_path/'tables/inference.csv'))\n"
            +"display(pd.read_csv(study_path/'tables/dv2_input_exposure_summary.csv'))\n"
            +"display(pd.read_csv(study_path/'tables/capacity_summary.csv'))\n"
            +"print(json.loads((study_path/'tables/accounting_audit.json').read_text()))"),
        nbformat.v4.new_code_cell(
            "for chart_name_str in ('equity_drawdown.png','fills_exposure.png','cost_sensitivity.png','rolling_correlation.png'):\n"
            +"    display(Image(filename=str(study_path/'charts'/chart_name_str)))"),
        nbformat.v4.new_markdown_cell("## הכרעה\n\nיש לקרוא את REPORT.md ואת מגבלות DV2 והתזמון. התאמה חשבונאית אינה ראיה למילוי בבורסה. המפרט, הקוד והנתונים קשורים בחתימות; כל 24 החלופות מוצגות.")
    ]
    with TemporaryDirectory(prefix="mr-dip-kernel-") as temp_str:
        kernel_path=Path(temp_str)/"research"
        kernel_path.mkdir()
        write_json(kernel_path/"kernel.json",{
            "argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
            "display_name":"Alpha research Python","language":"python"})
        manager_obj=KernelManager(kernel_name="research",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp_str]))
        client_obj=NotebookClient(notebook_obj,km=manager_obj,timeout=180,
            resources={"metadata":{"path":str(REPO_PATH)}})
        try:
            client_obj.execute()
        finally:
            if manager_obj.has_kernel:
                manager_obj.shutdown_kernel(now=True)
    notebook_obj.metadata["kernelspec"]={"display_name":"Python3 (select alpha_super .venv)","language":"python","name":"python3"}
    nbformat.write(notebook_obj,STUDY_PATH/"decision_notebook.ipynb")
    if any(output_obj.output_type=="error" for cell_obj in notebook_obj.cells
           if cell_obj.cell_type=="code" for output_obj in cell_obj.get("outputs",[])):
        raise AssertionError("Notebook execution error.")
    print("PASS executed decision notebook")


def finalize_lineage(verdict_str, any_cagr_improvement_bool, summary_df, delta_df):
    import importlib.util
    state_dict=json.loads((STUDY_PATH/"research_state.json").read_text(encoding="utf-8"))
    now_obj=datetime.now(timezone.utc)
    now_str=now_obj.isoformat()
    minutes_int=int((now_obj-datetime.fromisoformat(state_dict["created_at"])).total_seconds()/60.)
    if minutes_int>state_dict["runtime_budget"]["hard_cap_active_minutes"]:
        raise AssertionError("Research runtime exceeded declared hard budget.")
    registry_dict=json.loads((STUDY_PATH/"hypothesis_registry.json").read_text(encoding="utf-8"))
    helper_path=Path("C:/Users/User/.codex/skills/research-quant-signal-features/scripts/record_adaptive_event.py")
    module_spec=importlib.util.spec_from_file_location("event_helper",helper_path)
    helper_obj=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper_obj)
    receipt_list=[]
    for strategy_str in STRATEGY_TUPLE:
        result_path=STUDY_PATH/"runs"/strategy_str
        receipt_path=result_path/"runtime_acceleration.json"
        if receipt_path.exists():
            receipt_dict=json.loads(receipt_path.read_text())
            if receipt_dict["status"]!="completed" or not receipt_dict["full_native_parity"]:
                raise AssertionError("Metadata acceleration is incomplete.")
            if receipt_dict["wrapper_sha256"]!=sha256_file(REPO_PATH/"scripts/research/run_mr_deeper_dip_fast.py"):
                raise AssertionError("Metadata wrapper changed after execution.")
            for path_str,digest_str in receipt_dict["newly_executed_cells"].items():
                if sha256_file(result_path/path_str)!=digest_str:
                    raise AssertionError("Accelerated completion receipt drift.")
            receipt_list.append(str(receipt_path.relative_to(STUDY_PATH)))
        for hypothesis_str,layer_str,policy_list in (
                ("H0","central",["moo"]),
                ("H1","central",["limit_0.5pct","limit_1pct","limit_2pct"]),
                ("H2","stress",["moo","limit_0.5pct","limit_1pct","limit_2pct"])):
            experiment_str=strategy_str+"_"+hypothesis_str
            complete_list=[result_path/(layer_str+"_"+policy_str)/"complete.json" for policy_str in policy_list]
            identity_path=STUDY_PATH/"lineage_events"/(experiment_str+"_results.json")
            write_json(identity_path,{str(path_obj.relative_to(STUDY_PATH)):sha256_file(path_obj) for path_obj in complete_list})
            record_dict={
                "schema_version":"quant-research-experiment-v1","experiment_id":experiment_str,
                "study_id":state_dict["study_id"],"phase":"literal_baseline" if hypothesis_str=="H0" else "diagnosis",
                "recorded_at":now_str,"hypothesis_ids":[hypothesis_str],
                "declared_variant_count":len(policy_list),
                "data_periods_seen":["2010-01-05/2026-09-14; historical diagnostic; prior1998+ warmup"],
                "evidence_paths":[str(path_obj.relative_to(STUDY_PATH)) for path_obj in complete_list],
                "spec_content_id":"sha256:"+sha256_file(STUDY_PATH/"research_spec_frozen.json"),
                "code_content_id":"sha256:"+sha256_file(REPO_PATH/"scripts/research/run_mr_deeper_dip_study.py"),
                "result_content_id":"sha256:"+sha256_file(identity_path),
                "data_content_ids":["sha256:"+sha256_file(STUDY_PATH/"data"/strategy_str/"input_manifest.json")],
                "selection_role":strategy_str+"_"+layer_str+" predeclared reporting, no selected winner",
                "status":"completed",
            }
            record_path=STUDY_PATH/"lineage_events"/(experiment_str+".json")
            write_json(record_path,record_dict)
            helper_obj.append_record(STUDY_PATH,"experiment",record_path)
            next(hyp_dict for hyp_dict in registry_dict["hypotheses"]
                 if hyp_dict["hypothesis_id"]==hypothesis_str)["experiment_ids"].append(experiment_str)
    for hypothesis_dict in registry_dict["hypotheses"]:
        hypothesis_dict.update(status="completed",evidence_summary=(
            "Full source-native transactions/equity/cash parity for all3 systems."
            if hypothesis_dict["hypothesis_id"]=="H0" else verdict_str),
            disposition="replicated" if hypothesis_dict["hypothesis_id"]=="H0" else "diagnostic")
    registry_dict["updated_at"]=now_str
    write_json(STUDY_PATH/"hypothesis_registry.json",registry_dict)
    disposition_str="diagnostic"
    state_dict.update(updated_at=now_str,phase="complete",evidence_phase="diagnosis")
    state_dict["runtime_budget"].update(active_minutes_used=minutes_int,
        budget_stop_reason="All24 frozen economic cells, validation and artifacts completed.")
    state_dict["runtime_accounting_note"]="Elapsed wall minutes including waits used as a conservative active-runtime upper bound."
    state_dict["baseline"].update(status="completed",replication_outcome="replicated",
        executable_translation="Source-native next-open baselines replicated; limits conditional on dailyOHLC activation assumption.",
        primary_evidence=["tables/accounting_audit.json"]+receipt_list)
    state_dict["baseline"]["scope_note"]="Replication means3 native-code baselines. Article10daylow strategy not replicated by explicit scope waiver."
    state_dict["adaptive_search"].update(actual_total_variants=24,rounds_completed=0,
        stop_reason="All predeclared depths reported; no adaptive parameter search.")
    state_dict["final_decision"].update(disposition=disposition_str,research_status="diagnostic",
        verdict=verdict_str,next_gate="Predeclare an independent mechanism or verify causal intraday order activation; no operational promotion.")
    event_dict={"schema_version":"quant-research-decision-event-v1","event_id":"D0003",
        "study_id":state_dict["study_id"],"recorded_at":now_str,"phase":"complete",
        "decision":"close_frozen24_cell_experiment","reason":verdict_str,
        "evidence_paths":["REPORT.md","tables/summary.csv","tables/accounting_audit.json"],
        "holdout_consequence":"No untouched validation or confirmation period claimed.",
        "active_minutes_used":minutes_int}
    event_path=STUDY_PATH/"lineage_events/D0003.json"
    write_json(event_path,event_dict)
    source_dict=json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    knowledge_dict={
        "schema_version":"quant-research-knowledge-v1","study_id":state_dict["study_id"],
        "title":"Mean Reversion: Waiting for a Deeper Dip",
        "created_at":state_dict["created_at"],"last_reviewed_at":now_str,
        "research_status":"diagnostic","disposition":disposition_str,
        "replication_outcome":"replicated","replication_scope":"native code baselines only",
        "article_replication_outcome":"not_assessed","signal_family":"mean_reversion",
        "objective":source_dict["objective"],"verdict":verdict_str,
        "verdicts":{"source_replication":"3 native code baselines exactly replicated; separate article signal not attempted.",
                    "predictive_value":"Filled-only price improvement is conditional selection; all-opportunity and portfolio results reported.",
                    "economic_value":verdict_str,"promotion":"No operational promotion; no causal intraday fill evidence or unseen sample."},
        "universes":["DV2 native SP500 membership with retrospective5day tail exclusion",
                     "HPI235 exact historical SP500","Fixed11 US sector ETFs includingVOX/IYR"],
        "decision_timing":"Close_T","fill_timing":"Open_T+1 or conditional intradayDAYlimit",
        "timing_attribution":{"status":"tested","diagnostic_path":"Close_T to original exit model",
            "executable_path":"Open_T+1 to same exit",
            "method":"Per-opportunity compounded price decomposition; exit price includes native friction; excludes dividends/fees.",
            "headline_result":"Only the entry-signal-to-entry-open overnight component is excluded; later holding-period overnights remain included.",
            "metrics":json.loads(pd.read_csv(TABLE_PATH/"timing_attribution.csv").to_json(orient="records")),
            "artifact":"tables/timing_attribution.csv"},
        "primary_cost_layer":"central_research",
        "primary_metrics":{"period":"2010-01-05/2026-09-14","universe":"Three native universes",
            "cost_layer":"central and stress",
            "CAGR":None,"annualized_volatility":None,"Sharpe":None,"maximum_drawdown":None,"turnover":None,
            "aggregation_note":"No single primary portfolio or scalar metric; all24 independent cells below are reported without aggregating or selecting a winner.",
            "cells":json.loads(summary_df.to_json(orient="records"))},
        "inferential_results":{
            "method":"Predeclared HAC5 mean daily-return differences, Holm9 family correction",
            "scope":"Historical diagnostic; not a test of CAGR, Sharpe or drawdown differences",
            "comparisons":json.loads(pd.read_csv(TABLE_PATH/"inference.csv").to_json(orient="records"))},
        "feature_findings":[{"feature":row_obj.policy,"role":"entry execution",
            "direction":"central_cagr_higher" if row_obj.central_delta>0 else "central_cagr_lower",
            "status":"historical_diagnostic","effect_size":{"cagr_delta":row_obj.central_delta,"stress_delta":row_obj.stress_delta},
            "period_consistency":f"{row_obj.positive_periods}/4 positive central periods",
            "corrected_significance":"HAC5 +Holm9 in tables/inference.csv",
            "economic_mechanism":"Cheaper fills versus missed rebounds and idle capital.",
            "recommended_action":"No operational promotion; see full family.", "strategy":row_obj.strategy}
            for row_obj in delta_df.itertuples(index=False)],
        "cost_capacity":{"paper_like_round_trip_bps":None,
            "central_research_round_trip_bps":5.,"conservative_survival_round_trip_bps":25.,
            "bps_note":"Friction components only; per-share commissions with minimum1USD/order are additional. No full paper-like cost arm was run.",
            "capacity_impact_separate":True,"comfortable_capacity":None,"soft_capacity":None,
            "strained_capacity":None,"hard_capacity":None,"paper_like":"gross paired price decomposition, not literal article reconstruction",
            "central_round_trip":"5bp price/cash friction plus0.005USD/share eachside,minimum1USD/order",
            "conservative_round_trip":"25bp total modeledfriction plus commissions; limit5bp penetration requirement",
            "capacity_separate_from_base_cost":True,
            "capacity_labels":{"comfortable":None,"soft":None,"strained":None,"hard":None},
            "unresolved_reason":"Selected ADV63 demand at initial100k compounded equity only; no calibrated intraday impact or AUM sweep.",
            "artifact":"tables/capacity_summary.csv"},
        "limitations":[*source_dict["known_limits"],
            "Native whole-share rounding and per-share commissions use CAPITALSPECIAL-adjusted model units; contemporaneous actual shares were not reconstructed. Effect may differ between arms and is unmeasured."],
        "next_tests":["Only a predeclared independent mechanism or causal intraday execution test; do not tune more depths on this sample."],
        "sources":source_dict["sources"],"adaptive_lineage":{**state_dict["adaptive_search"],"profile":state_dict["profile"],"active_minutes_used":minutes_int},
        "artifacts":{**state_dict["artifacts"],"research_state":"research_state.json","notebook":"decision_notebook.ipynb",
            "primary_source_code":[str(REPO_PATH/"scripts/research"/name_str) for name_str in
                ("run_mr_deeper_dip_study.py","run_mr_deeper_dip_fast.py","analyze_mr_deeper_dip_study.py","report_mr_deeper_dip_study.py")],
            "primary_tables":["tables/summary.csv"],"primary_charts":["charts/equity_drawdown.png"]},
        "tags":["mean_reversion","entry_limit","DV2","HPI235","VOX","IYR","frozen24","historical_diagnostic"]}
    write_json(STUDY_PATH/"knowledge_record.json",knowledge_dict)
    validator_path=helper_path.with_name("validate_knowledge_record.py")
    validator_spec=importlib.util.spec_from_file_location("knowledge_validator",validator_path)
    validator_obj=importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator_obj)
    failure_list=validator_obj.validate_knowledge_record(STUDY_PATH/"knowledge_record.json")
    if failure_list:
        raise AssertionError(f"Knowledge schema errors: {failure_list}")
    write_json(STUDY_PATH/"research_state.json",state_dict)
    helper_obj.append_record(STUDY_PATH,"decision",event_path)
    print("PASS24-cell experiment lineage and validated knowledge record")


def update_local_knowledge_page():
    """Add an isolated Alpha study page; never rebuild the destructive shared KB."""
    import shutil
    page_path=REPO_PATH/"docs/research/alpha-studies/mr-deeper-dip-entry-study"
    page_path.mkdir(parents=True,exist_ok=True)
    for source_name_str,target_name_str in (("REPORT.md","index.md"),("REPORT_FULL.md","full.md"),
            ("DATA_AUDIT.md","DATA_AUDIT.md"),("SOURCE_RULE_MAP.md","SOURCE_RULE_MAP.md")):
        report_text_str=(STUDY_PATH/source_name_str).read_text(encoding="utf-8")
        report_text_str=report_text_str.replace("(REPORT_FULL.md)","(full.md)")
        report_text_str+="\n\n## חתימות העותק בבסיס הידע\n\n[חתימות העותק הזה](portal_manifest.json) מאמתות את הקבצים שהועתקו לעמוד. [חתימות המחקר המלא](run_manifest.json) מתייחסות לתיקיית המחקר המקורית, הכוללת גם נתונים והרצות שלא הועתקו לכאן.\n"
        (page_path/target_name_str).write_text(report_text_str,encoding="utf-8")
    for name_str in ("research_spec_frozen.json","research_spec_normalized_view.json","run_manifest.json","knowledge_record.json","decision_notebook.ipynb"):
        shutil.copy2(STUDY_PATH/name_str,page_path/name_str)
    for folder_str in ("charts","tables"):
        target_path=page_path/folder_str
        target_path.mkdir(exist_ok=True)
        for source_path in (STUDY_PATH/folder_str).iterdir():
            if source_path.is_file():
                shutil.copy2(source_path,target_path/source_path.name)
    portal_file_list=[path_obj for path_obj in sorted(page_path.rglob("*"))
                      if path_obj.is_file() and path_obj.name!="portal_manifest.json"]
    write_json(page_path/"portal_manifest.json",{
        "schema_version":"research-portal-copy-v1","created_at":datetime.now(timezone.utc).isoformat(),
        "source_study_path":str(STUDY_PATH),"source_manifest_sha256":sha256_file(STUDY_PATH/"run_manifest.json"),
        "files":[{"path":path_obj.relative_to(page_path).as_posix(),"bytes":path_obj.stat().st_size,
                  "sha256":sha256_file(path_obj)} for path_obj in portal_file_list]})
    config_path=REPO_PATH/"mkdocs.yml"
    config_str=config_path.read_text(encoding="utf-8")
    nav_line_str="      - Deeper Dip Entry Study: research/alpha-studies/mr-deeper-dip-entry-study/index.md\n"
    if nav_line_str not in config_str:
        anchor_str="  - Research:\n      - Overview: research/index.md\n"
        if config_str.count(anchor_str)!=1:
            raise AssertionError("Unexpected documentation navigation.")
        config_str=config_str.replace(anchor_str,anchor_str+nav_line_str)
    for exception_str in ("  !research/alpha-studies/\n","  !research/alpha-studies/**\n"):
        if exception_str not in config_str:
            config_str=config_str.replace("  !research/index.md\n","  !research/index.md\n"+exception_str)
    config_path.write_text(config_str,encoding="utf-8")
    print("PASS isolated local knowledge page",page_path)


def write_manifest():
    code_list=[
        "scripts/research/run_mr_deeper_dip_study.py",
        "scripts/research/run_mr_deeper_dip_fast.py",
        "scripts/research/analyze_mr_deeper_dip_study.py",
        "scripts/research/report_mr_deeper_dip_study.py",
        "tests/test_mr_deeper_dip_study.py"]
    exclude_set={"run_manifest.json"}
    file_dict={str(path_obj.relative_to(STUDY_PATH)):sha256_file(path_obj)
               for path_obj in sorted(STUDY_PATH.rglob("*"))
               if path_obj.is_file() and path_obj.name not in exclude_set}
    write_json(STUDY_PATH/"run_manifest.json",{
        "schema_version":"quant-research-manifest-v1","created_at":datetime.now(timezone.utc).isoformat(),
        "study_id":"mr_deeper_dip_entry_study","unique_economic_cells":24,
        "scope_note":"smoke/ and logs include technical tests/reruns; only runs/*/{central,stress}_*/complete.json are economic cells.",
        "self_exclusion":"run_manifest.json only",
        "files":[{"path":Path(name_str).as_posix(),"bytes":(STUDY_PATH/name_str).stat().st_size,
                  "sha256":digest_str} for name_str,digest_str in file_dict.items()],
        "external_deliverables_and_sources":[{"path":str(path_obj),"bytes":path_obj.stat().st_size,
            "sha256":sha256_file(path_obj)} for path_obj in
            [*[REPO_PATH/name_str for name_str in code_list],Path("C:/Users/User/Downloads/Where to Buy the (Deeper) Dip.pdf")]],
        "files_sha256":file_dict,
        "code_sha256":{name_str:sha256_file(REPO_PATH/name_str) for name_str in code_list}})
    print("PASS manifest",len(file_dict),"study files")

if __name__=="__main__":
    verdict_str,any_cagr_improvement_bool,summary_df,delta_df=build_reports()
    if "--closeout" in sys.argv:
        write_normalized_spec_view()
        write_notebook()
        finalize_lineage(verdict_str,any_cagr_improvement_bool,summary_df,delta_df)
        write_manifest()
        update_local_knowledge_page()
    print(verdict_str)
