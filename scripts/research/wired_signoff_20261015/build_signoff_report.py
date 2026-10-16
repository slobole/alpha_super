"""Publish the owner's compact Hebrew sign-off report from frozen audit evidence.

This tool only reads audit artifacts and source hashes. It cannot connect to a
provider/broker, change a release, approve capital or execute a trading command.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as xml_module
from datetime import UTC, datetime

import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))
ROOT_PATH = REPO_PATH / "results/review/wired_signoff_20261015"

STRATEGY_SPEC_LIST = [
    ("taa_rank_tqqq", "taa", "הקצאה מדורגת עם TQQQ", "strategy_taa_df_btal_fallback_tqqq_vix_cash"),
    ("taa_equal_tqqq", "taa", "הקצאה במשקל שווה עם TQQQ", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"),
    ("taa_linearity_qqq", "taa", "הקצאה לפי לינאריות עם QQQ", "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash"),
    ("ndx_atr", "ndx", "מומנטום נאסד״ק לפי ATR", "strategy_mo_atr_normalized_ndx"),
    ("ndx_atr_vxn", "ndx", "מומנטום נאסד״ק עם התאמת VXN", "strategy_mo_atr_normalized_ndx_vxn_scaled"),
    ("ndx_e2_atr", "ndx", "מומנטום E2 לפי ATR ומגבלת סקטור", "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap"),
    ("ndx_e2_natr20", "ndx", "מומנטום E2 לפי NATR20 ומגבלת סקטור", "strategy_mo_natr20_ndx_vxn_scaled_sector_cap"),
    ("dv2", "mr", "חזרה לממוצע במניות לפי DV2", "strategy_mr_dv2"),
    ("dv2_ind", "mr", "חזרה לממוצע בתעודות ענף לפי DV2", "strategy_mr_dv2_industry_etf"),
    ("hpi235", "mr", "חזרה לממוצע בהצבעת HPI 2/3/5", "strategy_mr_hpi_sp500_2_3_5_vote"),
    ("core5", "core5", "ליבה מאקרו אדפטיבית CORE5", "strategy_taa_adaptive_macro_core5"),
    *[(f"{parent_str}_{parking_str}", "capsules", f"קפסולת {parent_label_str} עם {parking_label_str}", f"strategy_mr_{module_parent_str}_vix_gated_{parking_str}")
      for parent_str, parent_label_str, module_parent_str in (("dv2", "DV2", "dv2"), ("hpi", "HPI", "hpi_vote"))
      for parking_str, parking_label_str in (("cash", "מזומן"), ("bil", "BIL"), ("spmo", "SPMO"))],
]
FAMILY_LABEL_DICT = {"taa": "הקצאת נכסים", "ndx": "מומנטום נאסד״ק", "mr": "חזרה לממוצע", "core5": "ליבה מאקרו", "capsules": "קפסולות חזרה לממוצע"}
FAMILY_EXPLANATION_DICT = {
    "taa": "חמש תעודות הגנה: GLD, UUP, TLT, DBC, BTAL. במשפחת המומנטום ממוצע תשואות של חודש, שלושה, שישה ושנים עשר חודשים נבחן מול סף DTB3 חודשי. משבצות שנכשלו עוברות ל־TQQQ; בגרסת הלינאריות ההקצאה נקבעת אחרת והחלופה היא QQQ. רכיב החלופה נשאר רק כאשר התנודתיות הממומשת של SPY נמוכה מ־VIX. הבדיקה חודשית, בסגירה האחרונה של החודש, לביצוע בפתיחה הבאה.",
    "ndx": "חברות נאסד״ק 100 לפי חברות היסטורית אמיתית. נדרשים SPY מעל ממוצע 200 ימים והמניה מעל ממוצע 100 ימים. הדירוג הוא תשואת שנים עשר חודשים ביחס ל־ATR נומינלי או ל־NATR20. עד עשר מניות, ללא דרישת תשואה חיובית. התאמת VXN מקטינה חשיפה; שני רכיבי E2 מוסיפים עד ארבע מניות לסקטור. כל רכיב מחובר לחשבון עצמאי; שילוב המחקר בחצי־חצי אינו העברת כספים בין חשבונות.",
    "mr": "שתי אסטרטגיות מניות משתמשות בחברות היסטורית במדד S&P 500; גרסת הענפים משתמשת ברשימה קבועה של תשע עשרה תעודות. ההחלטות יומיות בסגירה, עד עשר פוזיציות במשבצות של עשירית ההון, לביצוע בפתיחה הבאה. כללי הכניסה והיציאה שונים בין DV2 לבין HPI, ולכן נבדקו בנפרד.",
    "core5": "חמש משבצות של עשרים אחוז: SPY, IEF, GLD, DBC, UUP. כל משבצת עוברת בין הנכס לבין BIL לפי מגמה אדפטיבית. שכבת שורט ב־DBC תלויה בתנודתיות ומוגבלת לעשרה אחוזים בעת אירוע קביעת היעד. היעדים מחושבים באירוע שינוי מצב, בסוף חודש או באתחול; בדיקה יומית אינה איזון יומי. מספר המניות נקבע מסגירת היום, כולל השורט.",
    "capsules": "כללי DV2 או HPI נשמרים, ועל כניסות חדשות בלבד נוסף שער VIX משותף. יציאות אינן נחסמות בשער. לכל הורה שלוש זהויות: מזומן, BIL בלבד או SPMO עם BIL. החניה נקבעת בסוף שבוע מסחר או בשינוי השער; מכירות BIL למימון כניסות יכולות להתרחש גם ביניהם. לכל קפסולה נדרש חשבון ייעודי ותקציב חשבון מלא.",
}
FAMILY_FORMULA_DICT = {
    "taa": ["נוסחת המומנטום: m=(ROC1+ROC3+ROC6+ROC12)/4; h=(1+DTB3/100)^(1/12)-1; slot passes iff m>h.",
        "משקלי הדירוג: 5/15,4/15,3/15,2/15,1/15; equal slots=1/5.",
        "נוסחת הלינאריות: mean(beta_L*adjustedR2_L), L=21,63,126,252; adjustedR2=1-(1-R2)*(L-1)/(L-2); slot passes iff score>0.",
        "שער החלופה: RV20=std(SPY daily returns,ddof=0)*sqrt(252)*100; fallback allowed iff RV20<VIX.",
        "כמות חיה: floor(brokerNAV_preopen*pod_fraction*weight_T/referenceQuote_preopen); delta=target-actualShares."],
    "ndx": ["טווח אמיתי: TR=max(H-L,abs(H-Cprev),abs(L-Cprev)); ATR20_nominal=mean(TR_last20)*RawClose_T/AdjustedClose_T.",
        "נוסחת הדירוג: ROC12/ATR20_nominal; NATR20 version=ROC12/(ATR20_nominal/RawClose_T).",
        "סינון ומשקל: SPY>SMA200; stock>SMA100; top10, ties by symbol; each slot=0.10; unused slots stay cash.",
        "התאמת החשיפה בגרסאות VXN בלבד: scale=clip(22/VXN_asof_T,0.25,1); E2 accepts at most4 names per current-GICS sector, including UNKNOWN.",
        "כמות חיה: floor(brokerNAV_preopen*pod_fraction*weight_T/referenceQuote_preopen); E2 decision history=36 months, compared with full-history oracle."],
    "mr": ["נוסחת DV2: HL2=(H+L)/2; DV1=C/HL2-1; DV=mean(last2 DV1); DV2=100*average-tie percentile rank in last126 DV values, including T.",
        "כניסת DV2: DV2<10; Close>SMA200; Close/Close_(T-126)-1>0.05; rank NATR14 descending; exit Close_T>High_(T-1).",
        "סינון תעודות ענף: 252 observed sessions and ADV63=mean(native Turnover_last63)>50,000,000 USD; fixed19 ETF list.",
        "נוסחת HPI לכל אופק: Return_w=C_T/C_(T-w)-1; reference is previous1260 valid observations excluding T. For Return<=0:100*count(prior<=Return)/count(prior<=0); for Return>0:100*count(prior>Return)/count(prior>0).",
        "כניסת HPI: at least2 of horizons2/3/5 have Return<0 and HPI<30; IBS=(C-L)/(H-L)<0.10; Close>SMA200; rank native Turnover descending.",
        "יציאת HPI: IBS>0.90 or Wilder RSI2>90 or PIT membership loss; ten10%slots; persistent pending exits and same-open refill follow current stateful rules."],
    "core5": ["חומרת ירידה: d_T=1-P_TR,T/max(P_TR,s for s<=T); trailing126 midrank R=(N_less+(N_equal+1)/2)/126.",
        "ממוצע אדפטיבי: alpha=R^2*(2/51)+(1-R^2)*(2/201); AMA_T=alpha*P_TR,T+(1-alpha)*AMA_(T-1); first finite alpha initializes at P_TR,T.",
        "מצב לונג: L=1[SMA10_TR>AMA]; long weight=0.20*L; BIL weight=1-0.20*sum(L).",
        "מצב שורט: if SMA10_DBC<AMA_DBC, w_DBC=-min(0.10,0.025/(std63(returns,ddof=1)*sqrt252)); short proceeds stay cash. Short change alone is not a rebalance event.",
        "כמות חיה באירוע קביעת יעד: NAV_T=actual_cash_T+sum(actual_signed_shares*Close_T); target=trunc(NAV_T*event_weight/Close_T). Applied receipt preserves shares between events; mismatch generates ordinary daily catch-up.",
        "עלות השאלה במודל: abs(q_DBC)*ceil(1.02*Close_DBC)*0.01*calendar_days/360; terminal session does not accrue future borrow."],
    "capsules": ["שער הכניסה: threshold_T=mean(observed VIX since1990 throughT), minimum500 observations; open when VIX>threshold; close when VIX<=threshold after at least15 sessions from opening. Opening close is count0.",
        "כללי המניות: unchanged parent DV2 or HPI; exits always allowed; parking ETFs excluded from stock candidate list.",
        "סכום לחניה: idle=max(NAV_T-stock_value_after_intents_T-0.01*NAV_T,0); BIL rebalance band=0.01*NAV_T.",
        "משקל SPMO: min(1,0.08/(std20(daily returns,ddof=1)*sqrt252)); requires positive Volume on all last20 sessions; zero if insufficient or invalid.",
        "תזמון החניה: gate open =>SPMO0, idle inBIL; gate closed =>SPMO target at weekly close/gate transition, remainderBIL; between retargets BIL is sold if needed, proceeds from exits may wait.",
        "כמות חיה: stock entry dollars frozen atClose_T but shares=floor(entry_dollars/pre-submit quote); parking whole shares frozen atClose_T; actual holdings/cash and exact snapshot must refer to same session."],
}

HEBREW_FINDING_DICT = {
    "TAA-BTAL-HISTORY-11": ("עלויות BTAL בעבר אינן מבוססות למיליון דולר", "גם במשפחת ההקצאה נמצאה בעיית קיבולת היסטורית: בגרסה המדורגת הוראת BTAL ביולי 2017 שקולה לכ־330 אלף דולר מול חציון מחזור קודם של כ־19.5 אלף דולר — כמעט פי שבעה עשר. בשתי האחיות נמצאו הוראות בסדר גודל של פי עשרה מהמחזור, וחלק מההיסטוריה חסר מדד מחזור מספיק.", "חסם לחתימת ביצועי העבר: יש להציג את עלות הביצוע כהנחה בלתי מאומתת בתקופות האלה. שחזור מספרי זהה אינו מאשר שמילויי העבר היו זמינים בעלות המודל."),
    "TAA-BTAL-CURRENT-12": ("סחירות BTAL כיום מחייבת קבלה נפרדת", "בשלוש השנים האחרונות הוראות שקולות למיליון דולר הגיעו לכשישה אחוזים מהמחזור היומי. בשנה האחרונה לא היו הוראות BTAL במדגם. המחזור העדכני טוב מהעבר, אך חישוב לפי מחזור סוף התקופה הוא אבחון בלבד ולא הוכחת עומק מכרז.", "לפני הופעת הוראת BTAL יש לבחון את סכומה בפועל, מחיר הייחוס והסחירות הזמינה. אין להסיק קיבולת למיליון דולר רק ממחיר יומי או מהיעדר הוראות לאחרונה."),
    "MR-011": ("גם BIL אינו פותר את הנחת עלויות העבר", "בתחילת ההיסטוריה נמצאו בקפסולות BIL הוראות שקולות לכמאה עשרים ושישה אחוזים מהמחזור היומי ב־DV2 ולכשמונים ושניים אחוזים ב־HPI. בארבע הקפסולות עם חניה חסר חלון מחזור מלא להוראה מוקדמת אחת. הסחירות בתקופה האחרונה טובה בהרבה.", "יש לקבל את עלות העבר כקירוב בלתי מאומת גם בחלופת BIL. מחזור התעודה אינו מודד את כל נזילות נכסי הבסיס, ולכן אין כאן קביעה שהמילוי בלתי אפשרי."),
    "CORE5-CAPACITY-04": ("עלות BIL ההיסטורית בליבה היא קירוב", "בינואר 2008 הוראת BIL שקולה לכארבע מאות אלף דולר מול מחזור קודם של כ־2.79 מיליון דולר — כארבעה עשר אחוזים. בשלוש השנים האחרונות השיא המקביל היה כארבע מאיות האחוז בלבד.", "יש לתעד את שתי נקודה חמש נקודות הבסיס כהנחת מחקר שלא אומתה במכרזי העבר. נזילות נכסי הבסיס ויצירה ופדיון בתעודה עשויות לעלות על המחזור המוצג."),
    "TAA-DTB3-10": ("נתון הריבית ההיסטורי הוא קירוב למועד הפרסום", "שתי גרסאות המומנטום משתמשות בגרסת DTB3 הנוכחית ובמועד התצפית, לפי קירוב מחקר מוכר. בדחיית זמינות של יום מסחר אחד לא השתנה אף משקל במאה שישים ותשעה חודשים לכל גרסה; בקרות של חגים ונתון עתידי חריג זיהו את גבול הזמינות.", "אפשר לקבל את הקירוב עם המגבלה המתועדת. הבדיקה אינה הוכחה לגרסת הנתון שפורסמה בכל יום בעבר, ואינה פותרת את חסם VIX. גרסת הלינאריות אינה משתמשת ב־DTB3."),
    "MR-001": ("מסלולי DV2 ו־HPI הוותיקים ממשיכים אחרי פער באחזקות", "שוחזר במסלול בניית התכנית האמיתי: החלטה שחושבה מחשבון ריק פגשה חשבון עם עשר אחזקות. המערכת הזהירה ובנתה קנייה של מניה נוספת. הבדיקה נעצרה לפני שליחה; אין ראיה לפער בחשבון אמיתי.", "חסם להפעלת שני ההורים: יש לחסום תכנית שאחזקות הבסיס שלה שונות מאחזקות החשבון, ולבנות החלטה ממצב מהימן. נדרשת בדיקת רגרסיה במסלול החיצוני."),
    "MR-002": ("החלפת משבצת באותו מכרז תלויה בהצלחת המכירה", "הכללים מאפשרים קנייה באותה פתיחה שבה נשלחת יציאה. מכירה שנכשלה וקנייה שמולאה יכולות ליצור אחזקה או חשיפה עודפת; זו הנחה מוכרת ומאושרת.", "יש לקבל במפורש את ההנחה ולאמת הרשאות, חריגות, ביצוע חלקי והתאמה לחשבון אחרי הפריסה."),
    "MR-003": ("השער והחניה נבחרו אחרי ניסויים רבים", "נבחנו בערך מאה גרסאות שער ועשרה ענפי חניה באותה היסטוריה. תיקון סטטיסטי לתיק המשולב אינו הוכחה עצמאית לכל קפסולה.", "יש לחתום בנפרד על נכונות המימוש ועל אמונת התשואה, ולצבור תצפיות קדימה שלא שימשו לבחירה."),
    "MR-004": ("חניית SPMO מוסיפה סיכון מנייתי", "החניה נושאת חשיפת מניות כששער החזרה לממוצע סגור. הביצועים אינם תשואת בחירת מניות החזרה לממוצע לבדה; בהשוואות מחקר קודמות תוספת התשואה לוותה גם בהחמרת סיכון.", "יש להשוות כל קפסולת SPMO לאחותה עם BIL ולהפריד תשואה, ירידה מהשיא ומגבלות סחירות."),
    "MR-005": ("תעודות הענף הן רשימה נבחרת ומדגם מוגבל", "רשימת תשע עשרה התעודות נבחרה אחרי השוואת עשרים ושלושה יקומים וחשיפה לניסויים קודמים. מספרי המחקר הישנים שלפני תיקון Turnover אינם בסיס החתימה.", "יש להשתמש בשחזור המתוקן בדוח ולשמור את מסקנת המחקר כרשימת מעקב, גם כאשר בדיקות המימוש עוברות."),
    "MR-006": ("החשבונאות מחמירה ומקלה בכיוונים שונים", "ריבית אפס על מזומן חיובי וניכוי מס מדיבידנדים מחמירים לעומת מדד ברוטו; היעדר עלות מימון למזומן שלילי יכול להקל. זיכוי כלכלי ביום האקס אינו מועד התשלום בברוקר.", "יש לקרוא את מדידת יתרות המזומן ורגישות המימון לכל אסטרטגיה, ולפרק את פערי החשבונאות מול החשבון בפועל."),
    "MR-007": ("סטטוס היסטורי אינו אישור לגרסה הנוכחית", "מסמכים ישנים עדיין מציגים דרגות מחקר או הליכי התאוששות שכבר הוחלפו. בדיקות עבר שהמקור שלהן השתנה אינן מאשרות את הקוד הנוכחי.", "יש להסתמך על זהות האסטרטגיה, גרסת הקוד והראיות החדשות בכרטיס החתימה."),
    "MR-008": ("תשואת הכלל המלא אינה הוכחת יתרון בבחירת מניות", "מחקרי המקור דיווחו על יותר ממאתיים ניסויי DV2 ומאה ניסויי HPI. בדיקות אירועי הכניסה היו חלשות יותר מביצועי הכלל המלא, וב־DV2 נצפתה דעיכה. הנתונים האלה הם ראיות מחקר קודמות, ולא מבחן חדש בביקורת הזאת.", "יש לתאר את ההימור הכלכלי כשילוב סינון מגמה, חשיפה, יציאות ומשבצות; אין לראות בזהויות הקרובות מקורות תשואה בלתי תלויים."),
    "MR-009": ("עמלות ועיגול היסטוריים משתמשים במניות מתואמות", "מסלולי החזרה לממוצע מחשבים עיגול ועמלה לפי יחידות המחיר המתואם. פיצול עתידי יכול לשנות את התוצאה ההיסטורית; בדיקת אותות על מדגם מניות אינה מאשרת עמידות של החשבונאות.", "יש לקבל את תלות התוצאה בגרסת הנתונים. גודל ההשפעה הכולל לא נמדד מחדש ביחידות מניה היסטוריות גולמיות בביקורת הזאת."),
    "MR-010": ("עלויות חניית SPMO בעבר אינן מבוססות למיליון דולר", "שוחזרה קנייה באוקטובר 2017 השקולה לכ־626 אלף דולר בחשבון קבוע של מיליון, מול חציון מחזור קודם של כ־18.4 אלף דולר ליום — פי שלושים וארבעה. עשרים ימים עם מחזור חיובי לא מבטיחים קיבולת. הסחירות בשנה האחרונה טובה בהרבה; אין מכאן הוכחה לכשל מסחר כיום.", "חסם לחתימת עלויות העבר: יש לבסס עלות וקיבולת לתקופות הדלות, או להציג תקופה מצומצמת ומוגדרת מראש כהשערה חדשה. אין להכשיר את העבר לפי הסחירות כיום."),
    "TAA-HELPER-01": ("חסר בנתון VIX יכול לשנות החלטת הקצאה", "שוחזר בכל שלוש הגרסאות: מחיקת תצפית VIX אחרונה השאירה חישוב לפי תצפית קודמת ושינתה בתרחיש הבדיקה מאה אחוז מזומן למאה אחוז חלופה, בלי סימון גיל הנתון.", "נדרש אימות תצפיות העזר למועד ההחלטה, או מדיניות שימוש בנתון קודם שמאושרת ומוצגת במפורש."),
    "NDX-VXN-02": ("חשיפת נאסד״ק יכולה להיקבע לפי VXN ישן", "השימוש בתצפית קודמת הוא מדיניות קיימת ומאושרת. תאריך הסנאפשוט לבדו אינו מוכיח שגיל VXN תקין.", "מומלץ להציג תאריך תצפית וגיל בימי מסחר ב־Doctor ובדשבורד. החמרת המדיניות תהיה שינוי נפרד."),
    "MONTHLY-SIZING-03": ("התאמת משקלים אינה התאמת כמות או תשואה", "הבקטסט קובע כמות לפי סגירת היום; המסחר החודשי משתמש בהון ובמחיר ייחוס לפני הפתיחה. פער זה מכוון.", "נדרשת חתימה על חוזה הכמויות הזה, ורישום מחיר הייחוס, ההון והכמות שנשלחה."),
    "MONTHLY-ACCOUNTING-04": ("השוואת הייחוס החיה משתמשת בחשבונאות אחרת", "הבקטסט כולל דיבידנדים בניכוי עשרים וחמישה אחוזים; חלק מהשוואות הייחוס משתמשות במחיר בלבד. גם זמני הזיכוי בפועל שונים.", "נדרש לפרק פער תשואה לדיבידנדים, ריבית, עמלות, ביצוע ויתרה בלתי מוסברת לפני הסקת מסקנה על המימוש."),
    "TAA-UNITS-05": ("תוצאות TQQQ כוללות עיוות היסטורי ביחידות המניות", "עיגול ועמלות עדיין מבוססים על יחידות מחיר מתואמות. עיוות העמלות שנמדד בעבר היה בעיקר מחמיר; אין להסיק שכל השפעת העיגול מחמירה.", "החתימה מתייחסת למודל הקיים. מעבר ליחידות היסטוריות אמיתיות דורש בדיקה כמותית נפרדת; אומדן התיקון הישן אינו תוצאה חדשה."),
    "E2-SECTORS-06": ("סקטורי E2 היסטוריים מבוססים על סיווג נוכחי", "הסיווג נשמר ומאומת בקובץ, אך אינו משחזר את הסקטור שהיה ידוע בכל חודש היסטורי. מגבלת ארבע מניות גם אינה הבטחה לארבעים אחוזי חשיפה בפועל לאחר פערי פתיחה.", "אפשר לבחון את החיבור קדימה עם מגבלה גלויה; אין לחתום שהיסטוריית הסקטורים נקייה לחלוטין ממידע מאוחר."),
    "MONTHLY-EXECUTION-07": ("מחיר הפתיחה ההיסטורי אינו מחיר מכרז מאומת", "הספק מגדיר פתיחה כעסקה כשירה ראשונה בסרט המאוחד, מכל זירת מסחר. מחיר זה אינו מובטח להיות מחיר מכרז הפתיחה שיתקבל ב־MOO.", "נדרשות רגישות לעלויות, בדיקת נזילות וראיות מילוי בפועל לאחר חיבור מורשה."),
    "MONTHLY-SELECTION-08": ("היסטוריה שנבחנה פעמים רבות אינה מדגם חדש", "משפחת ההקצאה נבחרה מבין ארבעים ושמונה וריאנטים; לנאסד״ק תועדו מאה חמישים ואחת תצורות ועוד מחקרים חופפים. אין לחבר את הספירות אוטומטית.", "הביקורת משחזרת אסטרטגיות קיימות ללא אופטימיזציה. שחזור מוצלח אינו אישור לתשואה עודפת עתידית."),
    "NDX-HOST-09": ("בדיקת מועד חסרה בתוך מתאם נאסד״ק הישן", "המתאם הישיר יכול להחזיר תאריך ביצוע שכבר עבר, אך שערי המתזמן, בניית ההוראות, השליחה ו־Doctor חסמו את מסלול ההוראה שהתיישנה.", "זהו חיזוק הגנתי אפשרי, ולא פגיעת מסחר שהוכחה במסלול המוגן."),
    "CORE5-SELECTION-01": ("שכבת השורט בליבה לא עברה את כל מבחני המקור", "תועדו ארבע מאות ארבעים ואחד ניסויים ללא מדגם נקי. שכבת DBC נדחתה במבחן העברה המקורי ונשמרה בהחלטת הבעלים; מחקר עלות השאלה מאוחר יותר בחן שערים אחרים.", "נדרשת קבלה מפורשת של מגבלת הראיות. הצלחת מבחן עלות ההשאלה אינה מוחקת את הדחייה המקורית."),
    "CORE5-BORROW-02": ("עלות השאלה קבועה אינה זמינות שורט", "אחוז השאלה אחד לשנה הוא הנחת מחקר. קיימים סיכוני דחייה, שינוי עלות, החזרה כפויה וביצוע חלקי של סל ההוראות; אין בדיקת מימון מוקדמת לפי המדיניות המאושרת.", "לאחר פריסה מורשית יש לאמת חשבון, הרשאות, שורט וטיפול בחריגות בפועל. אין להחזיר בדיקות שהוסרו בלי שינוי מדיניות מפורש."),
    "CORE5-LEDGER-03": ("בליבה יש גם הנחות מחמירות וגם אופטימיות", "ניכוי דיבידנדים וריבית אפס על מזומן עשויים להחמיר; מימון מזומן שלילי אינו מחושב, ועלות השאלה קבועה אינה היסטוריה אמיתית.", "יש לקרוא את המספרים לצד מדיניות החשבונאות ורגישויות העלות, ולא לכנות את כל הבקטסט שמרני באופן גורף."),
    "OPS-DOCTOR-DAILY-01": ("בדיקת מחזור ב־Doctor אינה תואמת למחזור היומי החדש", "מחזור יומי שנשלח אמור להישאר פתוח עד סיום היום. שוחזר מצב שבו Doctor מסמן אותו כפג תוקף כבר במהלך המסחר.", "יש להתאים את האבחון למחזור היומי תוך שמירת החסימה על הוראה שלא נשלחה בזמן. אין להשתמש בחריגה כהוראת שליחה מחדש."),
    "OPS-DOCTOR-CORE5QUOTE-03": ("בדיקת הציטוט ב־Doctor מחמירה ממסלול CORE5 בפועל", "מסלול CORE5 משתמש במחיר סגירה קפוא ואינו מחייב ציטוט חי. שוחזר מצב שבו Doctor חסם בגלל אותו כשל ציטוט שבו המסלול האמיתי בנה תכנית תקינה.", "יש להתאים את בדיקת האבחון לחוזה המחיר הקפוא; אין להוסיף תלות ציטוט חדשה למסחר."),
    "OPS-DOC-02": ("מסמכי תפעול מכילים הוראות שהוחלפו", "נותרו תיאורי בדיקות מימון והשאלה והתאוששות ידנית, בעוד המדיניות הנוכחית משתמשת בקבלות ביצוע ובהשלמה רגילה ביום הבא.", "יש לתקן את מסמכי ההפעלה לפני מסירתם למפעיל, ולשמור את תוצאות העבר כשהן מסומנות כהיסטוריות."),
}


def read_json(path_obj: Path) -> dict:
    return json.loads(path_obj.read_text(encoding="utf-8"))


def hash_file(path_obj: Path) -> str:
    digest_obj = hashlib.sha256()
    with path_obj.open("rb") as handle_obj:
        for chunk_bytes in iter(lambda: handle_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def final_baseline_bool(summary_dict: dict) -> bool:
    return (summary_dict.get("status") == "pass" and summary_dict.get("deterministic_equity") is True
        and summary_dict.get("deterministic_economic_transactions") is True and summary_dict.get("capital_honored") is True
        and summary_dict.get("actual_capital_base") == 1_000_000 and summary_dict.get("actual_double_capital_base") == 2_000_000)


def execution_verified_bool(family_str: str, replay_dict: dict | None) -> bool:
    if not replay_dict:
        return False
    if family_str in {"taa", "ndx"}:
        return (replay_dict.get("mechanics_pass_bool") is True
            and replay_dict.get("informative_exact_int", 0) >= 24
            and replay_dict.get("exact_int", 0) == replay_dict.get("attempted_int", -1))
    if family_str == "core5":
        return (replay_dict.get("status_str") == "passed"
            and replay_dict.get("native_host_session_count_int", 0) >= 20
            and replay_dict.get("independent_nav_quote_invariance_case_count_int", 0) >= 4)
    session_count_int = replay_dict.get("session_count_int", 0)
    row_list = replay_dict.get("row_list", [])
    return (replay_dict.get("status_str") == "pass" and session_count_int >= 20
        and replay_dict.get("passed_session_count_int", 0) == session_count_int
        and replay_dict.get("coverage_dict", {}).get("with_order_intent", 0) >= 20
        and len(row_list) == session_count_int
        and all(row_dict.get("passed_bool") is True and row_dict.get("strategy_state_equal_bool") is True
            and row_dict.get("signal_close_timestamp_equal_bool") is True
            and row_dict.get("execution_timestamp_equal_bool") is True
            and row_dict.get("wrong_execution_date_positive_control_caught_bool") is True for row_dict in row_list))


def verify_saved_manifest(row_dict: dict) -> None:
    source_path = Path(row_dict["path"])
    if not source_path.is_absolute():
        source_path = REPO_PATH / source_path
    if not source_path.is_file() or hash_file(source_path) != row_dict["sha256"]:
        raise ValueError(f"Evidence changed: {source_path}; refresh the dependent evidence.")


def verify_replay_sources(replay_dict: dict, input_path: Path, replay_path: Path) -> None:
    """Require matching executed checker, frozen inputs and production sources."""
    for field_str, filename_str in (("input_sha256_str", "pricing.parquet"),
            ("auxiliary_sha256_str", "auxiliary.pkl.gz"), ("trace_sha256_str", "trace.pkl.gz")):
        verify_saved_manifest({"path": str(input_path / filename_str), "sha256": replay_dict[field_str]})
    for source_str, expected_str in replay_dict["source_hash_dict"].items():
        current_path = REPO_PATH / source_str
        if current_path.is_file() and hash_file(current_path) == expected_str:
            continue
        archive_path = replay_path.parent / "executed_replay_checker.py"
        if source_str.endswith("/mr_native_replay.py") and archive_path.is_file() and hash_file(archive_path) == expected_str:
            continue
        raise ValueError(f"Replay source changed without exact executed archive: {source_str}")
    if replay_dict.get("source_unchanged_bool") is not True:
        raise ValueError("Replay source was not unchanged at completion.")


def signoff_status_dict(variant_str: str, family_str: str, verified_bool: bool) -> dict:
    research_str = "ניתן לשחזר בכפוף להנחות"
    execution_str = "התאמה מותנית אומתה" if verified_bool else "חסרות ראיות"
    if family_str == "taa":
        research_str = "חסום: עלות BTAL היסטורית"
        execution_str = "חסום: אימות נתון עזר"
    if variant_str in {"dv2", "hpi235"}:
        execution_str = "חסום: פער אחזקות"
    if variant_str in {"dv2_spmo", "hpi_spmo"}:
        research_str = "חסום: עלות ביצוע היסטורית"
    if family_str == "core5":
        research_str = "מותנה בקבלת מגבלת השורט"
    return {"research_status": research_str, "execution_status": execution_str,
        "deployment_status": "טרם נבדק לאחר פריסה"}


def compact_liquidity_dict(liquidity_dict: dict) -> dict:
    """Retain totals, parking legs and largest tails; full source stays linked."""
    compact_dict = copy.deepcopy(liquidity_dict)
    details_dict = compact_dict.get("period_details", {})
    asset_list = details_dict.get("asset_list", [])
    selected_set = {"BIL", "SPMO"}
    for period_str, metric_str, count_int in (("last3y", "p99", 5), ("full", "max", 3)):
        ranked_list = sorted(asset_list, key=lambda row_dict: row_dict.get(period_str, {}).get("constant_1m_available_subset", {}).get(metric_str) or 0, reverse=True)
        selected_set.update(row_dict["asset"] for row_dict in ranked_list[:count_int])
    details_dict["asset_detail_count_total"] = len(asset_list)
    details_dict["asset_detail_selection"] = "BIL/SPMO plus largest5 recent p99 and largest3 full-history maxima; complete per-asset evidence is in the linked statistics file"
    details_dict["asset_list"] = [row_dict for row_dict in asset_list if row_dict["asset"] in selected_set]
    return compact_dict


def collect_tests() -> dict:
    ignored_set = {"mr_offline_tests.xml", "taa_ndx_focused_tests.xml"}
    case_status_dict = {}
    skipped_case_dict = {}
    evidence_list = []
    for xml_path in sorted(ROOT_PATH.glob("*.xml")):
        if xml_path.name in ignored_set:
            continue
        root_obj = xml_module.parse(xml_path).getroot()
        node_list = root_obj.findall(".//testcase")
        for node_obj in node_list:
            nodeid_str = node_obj.attrib.get("classname", "") + "::" + node_obj.attrib["name"]
            status_str = "failed" if node_obj.find("failure") is not None or node_obj.find("error") is not None else "skipped" if node_obj.find("skipped") is not None else "passed"
            prior_str = case_status_dict.get(nodeid_str)
            case_status_dict[nodeid_str] = "failed" if "failed" in (prior_str, status_str) else "passed" if "passed" in (prior_str, status_str) else "skipped"
            if status_str == "skipped":
                skipped_case_dict[nodeid_str] = node_obj.find("skipped").get("message", "")
        evidence_list.append({"path": str(xml_path.relative_to(ROOT_PATH)), "sha256": hash_file(xml_path), "cases": len(node_list)})
    return {"passed": sum(value_str == "passed" for value_str in case_status_dict.values()),
        "failed": sum(value_str == "failed" for value_str in case_status_dict.values()),
        "skipped": sum(value_str == "skipped" for value_str in case_status_dict.values()), "xml": evidence_list,
        "skipped_reasons": {key_str: value_str for key_str, value_str in skipped_case_dict.items() if case_status_dict[key_str] == "skipped"},
        "node_status": case_status_dict, "deduplication": "classname::testname; any failure in selected final evidence dominates; skipped is never a pass"}


def verify_production_sources(manifest_dict: dict) -> None:
    changed_list = [row_dict["path"] for row_dict in manifest_dict["source_files"] if hash_file(REPO_PATH / row_dict["path"]) != row_dict["sha256"]]
    if changed_list:
        raise ValueError(f"Production source changed after audit freeze: {changed_list}")



def apply_owner_review(report_dict: dict, owner_review_dict: dict) -> dict:
    """Apply scoped owner decisions to presentation; retain all original evidence."""
    if owner_review_dict.get("schema") != "wired_signoff.owner_review.v1":
        raise ValueError("Unknown owner-review schema.")
    if report_dict["head"] != owner_review_dict["audit_head"]:
        raise ValueError("Owner review belongs to a different frozen audit.")
    finding_by_id_dict = {item_dict["id"]: item_dict for item_dict in report_dict["findings"]}
    card_by_variant_dict = {item_dict["variant"]: item_dict for item_dict in report_dict["cards"]}
    decision_list = owner_review_dict["decisions"]
    decision_id_list = [item_dict["id"] for item_dict in decision_list]
    if len(decision_id_list) != len(set(decision_id_list)):
        raise ValueError("Duplicate owner decision.")
    if set(decision_id_list) - finding_by_id_dict.keys():
        raise ValueError("Owner decision references an unknown finding.")
    if set(owner_review_dict["card_updates"]) - card_by_variant_dict.keys():
        raise ValueError("Owner review references an unknown strategy.")
    allowed_status_set = {"accepted", "partially_accepted", "deferred", "historical_disclosure"}
    for decision_dict in decision_list:
        if decision_dict["status"] not in allowed_status_set:
            raise ValueError("Unknown disposition; owner acceptance is not a technical fix.")
        if decision_dict["status"] == "partially_accepted":
            accepted_variant_set = set(decision_dict.get("accepted_variant_list", []))
            open_variant_set = set(decision_dict.get("open_variant_list", []))
            affected_variant_set = set(finding_by_id_dict[decision_dict["id"]]["affected"])
            if (not accepted_variant_set or not open_variant_set
                    or accepted_variant_set & open_variant_set
                    or accepted_variant_set | open_variant_set != affected_variant_set):
                raise ValueError("Partial acceptance must partition the original finding scope.")
    for remaining_dict in owner_review_dict["remaining_items"]:
        if remaining_dict["priority"] not in {1, 2, 3} or set(remaining_dict["finding_ids"]) - finding_by_id_dict.keys():
            raise ValueError("Invalid remaining action or source finding.")
        if set(remaining_dict.get("affected", [])) - card_by_variant_dict.keys():
            raise ValueError("Remaining action references an unknown strategy.")
    reviewed_report_dict = copy.deepcopy(report_dict)
    reviewed_finding_dict = {item_dict["id"]: item_dict for item_dict in reviewed_report_dict["findings"]}
    for decision_dict in decision_list:
        reviewed_finding_dict[decision_dict["id"]]["owner_disposition"] = copy.deepcopy(decision_dict)
    for card_dict in reviewed_report_dict["cards"]:
        update_dict = owner_review_dict["card_updates"].get(card_dict["variant"])
        if not update_dict:
            continue
        card_dict.setdefault("audit_status_before_owner_review", {
            field_str: card_dict[field_str] for field_str in ("research_status", "execution_status", "conditions")
        })
        baseline_dict = card_dict["baseline_evidence"]
        baseline_verified_bool = (
            baseline_dict.get("deterministic_equity") is True
            and baseline_dict.get("deterministic_economic_transactions") is True
            and baseline_dict.get("actual_capital_base") == 1_000_000
            and baseline_dict.get("actual_double_capital_base") == 2_000_000
        )
        if "research_status" in update_dict:
            card_dict["research_status"] = update_dict["research_status"] if baseline_verified_bool else "חסרות ראיות לשחזור"
        if "execution_status" in update_dict:
            replay_verified_bool = execution_verified_bool(card_dict["family"], card_dict.get("replay"))
            card_dict["execution_status"] = update_dict["execution_status"] if replay_verified_bool else "חסרות ראיות"
        prefix_tuple = tuple(update_dict["remove_condition_prefixes"])
        condition_list = [item_str for item_str in card_dict["conditions"] if not item_str.startswith(prefix_tuple)]
        card_dict["conditions"] = list(dict.fromkeys([*update_dict["add_conditions"], *condition_list]))
    if "same_auction_followup" in owner_review_dict:
        followup_dict = copy.deepcopy(owner_review_dict["same_auction_followup"])
        reviewed_report_dict["same_auction_followup"] = followup_dict
        reviewed_finding_dict["MR-002"]["operational_followup"] = copy.deepcopy(followup_dict)
        accepted_variant_set = set(followup_dict["accepted_daily_variant_list"])
        legacy_variant_set = set(followup_dict["open_legacy_variant_list"])
        if (accepted_variant_set & legacy_variant_set
                or (accepted_variant_set | legacy_variant_set) - card_by_variant_dict.keys()):
            raise ValueError("Daily acceptance and open legacy scopes must be known and disjoint.")
        for card_dict in reviewed_report_dict["cards"]:
            if card_dict["variant"] in accepted_variant_set:
                card_dict["same_auction_signoff_status"] = "accepted_residual_daily_risk"
            elif card_dict["variant"] in legacy_variant_set:
                card_dict["same_auction_signoff_status"] = "open_legacy_risk"
    if "signoff_snapshot" in owner_review_dict:
        reviewed_report_dict["signoff_snapshot"] = copy.deepcopy(owner_review_dict["signoff_snapshot"])
    reviewed_report_dict.setdefault("evidence_generated_utc", report_dict["generated_utc"])
    reviewed_report_dict["generated_utc"] = owner_review_dict["recorded_utc"]
    reviewed_report_dict["owner_review"] = copy.deepcopy(owner_review_dict)
    reviewed_report_dict["remaining_items"] = copy.deepcopy(owner_review_dict["remaining_items"])
    reviewed_report_dict["operator_improvements"] = owner_review_dict["operator_improvements"]
    reviewed_report_dict["summary"] = owner_review_dict["summary"]
    return reviewed_report_dict


def write_report_artifacts(report_dict: dict) -> None:
    """Render the same report data in each existing presentation format."""
    payload_str = json.dumps(report_dict, ensure_ascii=False, allow_nan=False)
    html_template_str = (Path(__file__).parent / "report_view.html").read_text(encoding="utf-8")
    canvas_template_str = (Path(__file__).parent / "report_canvas.template").read_text(encoding="utf-8")
    (ROOT_PATH / "audit_report.json").write_text(payload_str, encoding="utf-8")
    (ROOT_PATH / "signoff_report.html").write_text(html_template_str.replace("__AUDIT_JSON__", payload_str.replace("</", "<\\/")), encoding="utf-8")
    (ROOT_PATH / "wired-signoff.canvas.tsx").write_text(canvas_template_str.replace("__AUDIT_JSON__", payload_str), encoding="utf-8")


def main() -> None:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--preview", action="store_true")
    parser_obj.add_argument("--owner-review-only", action="store_true", help="Update presentation decisions without rerunning or recertifying audit evidence.")
    args_obj = parser_obj.parse_args()
    owner_review_path = ROOT_PATH / "owner_review_20261016/owner_decisions.json"
    if args_obj.owner_review_only:
        report_dict = apply_owner_review(read_json(ROOT_PATH / "audit_report.json"), read_json(owner_review_path))
        write_report_artifacts(report_dict)
        print(json.dumps({"owner_review_date": report_dict["owner_review"]["review_date"], "cards": len(report_dict["cards"]), "findings": len(report_dict["findings"]), "remaining_groups": len(report_dict["remaining_items"]), "evidence_rerun": False}))
        return
    manifest_dict = read_json(ROOT_PATH / "frozen_manifest.json")
    verify_production_sources(manifest_dict)
    review_dict = {key_str: read_json(ROOT_PATH / filename_str) for key_str, filename_str in
        (("monthly", "taa_ndx_review.json"), ("core5", "core5_review.json"), ("ops", "ops_review.json"))}
    if (ROOT_PATH / "mr_review.json").exists():
        review_dict["mr"] = read_json(ROOT_PATH / "mr_review.json")
    for family_str in ("monthly", "core5"):
        for artifact_dict in review_dict[family_str]["artifact_manifest_list"]:
            verify_saved_manifest(artifact_dict)
    for family_str in ("taa", "ndx"):
        replay_path = ROOT_PATH / "taa_ndx_native" / f"{family_str}_actual_host_replay.json"
        family_replay_dict = read_json(replay_path)
        verify_saved_manifest({"path": str(Path(__file__).parent / "taa_ndx_native_replay.py"), "sha256": family_replay_dict["script_sha256_str"]})
        if family_str == "taa":
            for relative_str, field_str in (("scripts/research/wired_signoff_20261015/taa_ndx_taa_frozen_capture.py", "capture_script_sha256_str"),
                    ("results/review/wired_signoff_20261015/inputs/DTB3.csv", "fred_input_sha256_str"),
                    ("results/review/wired_signoff_20261015/inputs/native_loader_cache.pkl.gz", "frozen_native_cache_sha256_str"),
                    ("results/review/wired_signoff_20261015/frozen_manifest.json", "frozen_source_manifest_sha256_str")):
                verify_saved_manifest({"path": relative_str, "sha256": family_replay_dict[field_str]})
            for artifact_dict in family_replay_dict.get("native_input_manifest_list", []):
                verify_saved_manifest({"path": artifact_dict["path_str"], "sha256": artifact_dict["sha256_str"]})
            if not family_replay_dict.get("source_unchanged_bool") or not family_replay_dict.get("input_files_unchanged_bool"):
                raise ValueError("TAA requires captured input/source provenance, not only replay counts.")
        else:
            snapshot_path = REPO_PATH / "results/review/e2_memory_20261014/refresh_snapshot_root/norgate_eod_ndx_e2_pit_plus_vxn_helper/2026-09-30"
            verify_saved_manifest({"path": str(snapshot_path / "manifest.json"), "sha256": family_replay_dict["source_manifest_hash_str"]})
            snapshot_dict = read_json(snapshot_path / "manifest.json")
            for filename_str in ("prices.parquet", "universe.parquet"):
                verify_saved_manifest({"path": str(snapshot_path / filename_str), "sha256": snapshot_dict["files"][filename_str]["sha256"]})
    core_replay_dict = review_dict["core5"]["current_code_saved_input_native_replay_dict"]
    verify_saved_manifest(core_replay_dict["artifact_dict"])
    verify_saved_manifest({"path": core_replay_dict["prices_path_str"], "sha256": core_replay_dict["prices_sha256_str"]})
    for source_str, expected_str in core_replay_dict["source_sha256_dict"].items():
        verify_saved_manifest({"path": source_str, "sha256": expected_str})
    bridge_dict = read_json(ROOT_PATH / "monthly_canonical_trace_bridge.json")
    if bridge_dict.get("status") != "pass" or bridge_dict.get("attempted") != 265 or bridge_dict.get("passed") != 265:
        raise ValueError("Fresh canonical monthly engine traces do not match all265 host replay decisions.")
    for source_dict in [*bridge_dict["input_manifest"], bridge_dict["source"]]:
        verify_saved_manifest(source_dict)
    test_dict = collect_tests()
    if test_dict["failed"]:
        raise ValueError("Selected final test evidence contains failures; do not publish a clean signoff.")
    node_path = ROOT_PATH / "test_node_status.json"
    node_path.write_text(json.dumps(test_dict.pop("node_status"), indent=2), encoding="utf-8")
    test_dict["node_evidence"] = {"path": node_path.name, "bytes": node_path.stat().st_size, "sha256": hash_file(node_path)}
    raw_finding_list = [*review_dict["monthly"]["findings_list"], *review_dict["core5"]["research_caveat_list"], *review_dict["ops"]["findings_list"]]
    raw_finding_list.extend(review_dict.get("mr", {}).get("findings", []))
    finding_list = []
    for item_dict in raw_finding_list:
        identifier_str = item_dict.get("id_str", item_dict.get("id"))
        title_str, body_str, action_str = HEBREW_FINDING_DICT.get(identifier_str, ("ממצא נוסף בביקורת", "פירוט הממצא והראיות נמצא בנספח המקורות.", "נדרש לקרוא את ההמלצה הטכנית ואת האסטרטגיות המושפעות לפני חתימה."))
        priority_str = item_dict.get("priority_str", item_dict.get("priority", "P2"))
        affected_list = item_dict.get("affected_variant_list", item_dict.get("strategies", []))
        if identifier_str.startswith("CORE5-") or identifier_str == "OPS-DOCTOR-CORE5QUOTE-03":
            affected_list = ["core5"]
        if identifier_str == "OPS-DOCTOR-DAILY-01":
            affected_list = ["core5", "dv2_ind", *[row_tuple[0] for row_tuple in STRATEGY_SPEC_LIST if row_tuple[1] == "capsules"]]
        finding_list.append({"id": identifier_str, "priority": int(priority_str[1]) if priority_str.startswith("P") else 2,
            "priority_detail": priority_str, "title": title_str, "body": body_str, "action": action_str,
            "confidence": item_dict.get("confidence_str", item_dict.get("confidence", "source_verified")), "affected": affected_list,
            "sources": item_dict.get("source_list", item_dict.get("evidence", [])), "technical": item_dict})
    action_order_dict = {"MR-001": 0, "TAA-HELPER-01": 1, "TAA-BTAL-HISTORY-11": 2, "MR-010": 3,
        "CORE5-SELECTION-01": 4, "CORE5-BORROW-02": 5,
        "OPS-DOCTOR-DAILY-01": 5, "OPS-DOCTOR-CORE5QUOTE-03": 6}
    finding_list.sort(key=lambda item_dict: (item_dict["priority"], action_order_dict.get(item_dict["id"], 50), item_dict["id"]))
    cards_list = []
    pending_list = []
    for variant_str, family_str, title_str, key_str in STRATEGY_SPEC_LIST:
        baseline_path = ROOT_PATH / "baselines" / key_str
        summary_dict = read_json(baseline_path / "summary.json") if (baseline_path / "summary.json").exists() else {}
        statistic_path = ROOT_PATH / "audit_statistics" / key_str / "statistics.json"
        if not final_baseline_bool(summary_dict) or not statistic_path.exists():
            pending_list.append(key_str)
            continue
        statistic_dict = read_json(statistic_path)
        for source_dict in [*statistic_dict["input_manifest"], statistic_dict["statistics_source"],
                            *statistic_dict["chart_data_manifest"]]:
            verify_saved_manifest(source_dict)
        if statistic_dict.get("upstream_source_manifest"):
            verify_saved_manifest(statistic_dict["upstream_source_manifest"])
        daily_df = pd.read_csv(statistic_path.parent / "daily_chart.csv.gz", index_col=0, parse_dates=True)
        # *** CRITICAL *** Chart sampling is descriptive only. Statistics and
        # MaxDD remain based on the complete DAILY path; chart labels disclose
        # monthly points and monthly minimum drawdown, including partial MTD.
        sampled_df = daily_df.resample("ME").last()
        sampled_df["drawdown"] = daily_df["drawdown"].resample("ME").min()
        chart_columns = [column_str for column_str in ("SPX_TR_strategy_common_close_equity", "SPX_TR_benchmark_common_close_equity", "drawdown", "SPX_TR_rolling126_correlation") if column_str in sampled_df]
        point_list = [{"date": str(min(date_ts, daily_df.index[-1]).date()), **{column_str: None if pd.isna(row_ser[column_str]) else float(row_ser[column_str]) for column_str in chart_columns}} for date_ts, row_ser in sampled_df.iterrows()]
        monthly_variant_dict = next((row_dict for row_dict in review_dict["monthly"]["variant_review_list"] if row_dict["variant_str"] == variant_str), {})
        replay_dict = monthly_variant_dict.get("fresh_actual_host_replay_dict")
        if family_str == "core5":
            replay_dict = review_dict["core5"].get("current_code_saved_input_native_replay_dict")
        if family_str in {"mr", "capsules"}:
            replay_artifact_str = next((row_dict["artifact"] for row_dict in review_dict.get("mr", {}).get("native_replays_completed", []) if row_dict["variant"] == variant_str), "")
            replay_path = ROOT_PATH / replay_artifact_str if replay_artifact_str else ROOT_PATH / f"mr_{variant_str}" / "replay.json"
            replay_dict = read_json(replay_path) if replay_path.exists() else None
            if replay_dict and replay_dict.get("status_str") == "pass":
                replay_input_path = ROOT_PATH / "mr_hpi235_trace_only" if variant_str == "hpi235" else baseline_path
                verify_replay_sources(replay_dict, replay_input_path, replay_path)
        verified_bool = execution_verified_bool(family_str, replay_dict)
        if not verified_bool:
            pending_list.append(f"{key_str}: execution replay")
        condition_list = ["תוצאות המודל ניתנות לשחזור; אין מדגם חדש שלא נבחן בעבר.",
            "תנאי החשבון, הכמויות ומדיניות העלויות דורשים קבלה מפורשת לפני הפעלה.",
            "בדיקת קבלה בסביבת הפריסה וביצועים אמיתיים טרם בוצעו בביקורת הזאת."]
        if family_str == "taa":
            condition_list.insert(0, "חסם לחתימה מלאה: יש לטפל באימות תצפית VIX או לאשר מדיניות שימוש קודמת גלויה.")
            condition_list.insert(1, "חסם במחקר: עלויות BTAL בתקופות מחזור דל לא הוכחו למיליון דולר. בעת הוראה חדשה נדרשת גם קבלת סחירות לפי סכום ההוראה בפועל.")
            if variant_str != "taa_linearity_qqq":
                condition_list.append("קירוב מחקר מאושר: גרסת DTB3 נוכחית ומועד תצפית. דחייה של יום מסחר אחד לא שינתה משקלים במאה שישים ותשעה חודשים; זו אינה הוכחת פרסום היסטורית.")
        if family_str == "ndx" and "e2" in variant_str:
            condition_list.insert(0, "מגבלה היסטורית: סיווגי הסקטורים נוכחיים ולא היסטוריים לפי מועד הידיעה.")
        if family_str == "core5":
            condition_list.insert(0, "מגבלת מחקר מהותית: שכבת השורט נשמרה למרות דחייה במבחן ההעברה המקורי.")
            condition_list.append("מגבלת עלויות: הוראות BIL גדולות ביחס למחזור בתחילת ההיסטוריה; קבלת עלות המודל אינה הוכחת עלות מכרז בפועל.")
        if variant_str in {"dv2", "hpi235"}:
            condition_list.insert(0, "חסם מימוש: פער בין אחזקות בסיס ההחלטה לאחזקות החשבון חייב לחסום שליחה. המסלול הנוכחי מתריע וממשיך.")
        if variant_str in {"dv2_spmo", "hpi_spmo"}:
            condition_list.insert(0, "חסם מחקר: עלויות וקיבולת בתקופות סחירות דלות לא הוכחו למיליון דולר. סחירות עדכנית טובה יותר אינה מאמתת את מילויי העבר.")
        if variant_str == "dv2_ind":
            condition_list.insert(0, "מסקנת המחקר נותרת רשימת מעקב: רשימת תעודות קבועה שנבחרה בדיעבד, והיסטוריה מוגבלת. יש להשתמש רק במחזור הדולרי המתוקן.")
        if family_str in {"mr", "capsules"}:
            condition_list.append("מגבלת חשבונאות: מניות מתואמות משמשות לעיגול ולעמלה; ההשפעה המלאה של מעבר ליחידות גולמיות לא נמדדה מחדש.")
        if family_str == "capsules":
            condition_list.append("מגבלת בחירה: כמאה ניסויי שער ועשרה ענפי חניה על היסטוריה חופפת; לכל זהות דרוש חשבון ייעודי.")
            if not variant_str.endswith("cash"):
                condition_list.append("מגבלת חניה: גם עלות BIL בהיסטוריה הדלה נשארת קירוב; חלופת BIL אינה מתקנת לבדה את תוקף מילויי העבר.")
        source_summary_dict = {"statistics": str(statistic_path.relative_to(ROOT_PATH)).replace("\\", "/"),
            "statistics_sha256": hash_file(statistic_path), "replay_qualified": verified_bool}
        status_dict = signoff_status_dict(variant_str, family_str, verified_bool)
        if family_str in {"taa", "ndx"}:
            execution_contract_str = "חוזה הכמות: משקלי היעד נשמרים מסגירת ההחלטה. לפני הפתיחה מספר המניות מחושב מהון הברוקר כפול תקציב המסלול, חלקי מחיר הייחוס העדכני, ובעיגול מטה. התאמת המשקלים אינה התאמת מניות או תשואה."
        elif family_str == "core5":
            execution_contract_str = "חוזה הכמות: מניות שלמות חתומות נקבעות מהון ומחירי סגירת ההחלטה, ונשמרות בקבלת היעד בין אירועים. הון וציטוט מאוחרים אינם משנים את היעד הקפוא."
        elif family_str == "capsules":
            execution_contract_str = "חוזה הכמות: סכום כניסת מניה נשמר מעשירית הון סגירת ההחלטה ומומר למניות לפי ציטוט השליחה. לחניה נשמר יעד מניות שלמות מסגירת ההחלטה. החשבון צריך להיות עצמאי ובתקציב מלא."
        elif variant_str == "dv2_ind":
            execution_contract_str = "חוזה הכמות: הון סגירת ההחלטה מחושב ממזומן ומניות פיזיות במחיר סגירה גולמי. עשירית ממנו נשמרת לכל כניסה ומומרת למניות לפי ציטוט השליחה. השחזור אימת דירוג, יציאות ומשקל עשירית; הוא אינו הוכחת מניות פיזיות היסטוריות."
        else:
            execution_contract_str = "חוזה הכמות בהורים הוותיקים: כוונת כניסה של עשירית ההון מומרת למשקל. המימוש קובע מניות לפי עשירית הון הברוקר העדכני כפול תקציב המסלול וחלקי ציטוט השליחה, בעיגול מטה. אחזקות שלא נבחרו ליציאה נשמרות; המסלול הזה שונה מחוזה ההון הקפוא בקפסולות."
        cards_list.append({"variant": variant_str, "key": key_str, "family": family_str, "title": title_str,
            "tier": next(row_dict["tier"] for row_dict in manifest_dict["strategies"] if key_str == row_dict["import"].split(":", 1)[0].rsplit(".", 1)[-1]),
            "metrics": statistic_dict["standalone_metrics"], "benchmark": statistic_dict["benchmarks"].get("SPX_TR", {}),
            "subperiods": statistic_dict["subperiods"], "liquidity": compact_liquidity_dict(statistic_dict["liquidity"]), "costs": statistic_dict["cost_sensitivities"],
            "accounting": summary_dict.get("accounting_policy", {}),
            "config": summary_dict.get("module_default_config") if isinstance(summary_dict.get("module_default_config"), dict) else None,
            "baseline_evidence": {name_str: summary_dict.get(name_str) for name_str in ("deterministic_equity", "deterministic_economic_transactions", "double_capital_ratio", "actual_capital_base", "actual_double_capital_base")},
            **status_dict, "replay": {field_str: value_obj for field_str, value_obj in (replay_dict or {}).items() if field_str != "row_list"},
            "execution_contract": execution_contract_str,
            "cash": {"negative_cash_days": summary_dict.get("negative_cash_days"),
                "min_cash_fraction": summary_dict.get("min_cash_fraction"), "commission_total": summary_dict.get("commission_total")},
            "evidence_summary": source_summary_dict,
            "conditions": condition_list, "chart": point_list,
            "evidence": str(statistic_path.relative_to(ROOT_PATH)).replace("\\", "/")})
    if pending_list and not args_obj.preview:
        raise ValueError(f"Audit incomplete; no final report published: {pending_list}")
    report_dict = {"schema": "wired_signoff.report.v1", "generated_utc": datetime.now(UTC).isoformat(),
        "title": "ביקורת וחתימה לאסטרטגיות המחוברות", "head": manifest_dict["head"],
        "data_endpoint": "2026-10-14", "model_budget": 1_000_000, "complete": not pending_list,
        "summary": "הביקורת מפרידה בין אמינות התוצאות, נאמנות ההחלטות והביצוע, וקבלה בסביבת המסחר. לכל אסטרטגיה חתימה נפרדת עם תנאים וראיות; אין שינוי בהקצאה, בפריסה או במסחר.",
        "family_labels": FAMILY_LABEL_DICT, "family_explanations": FAMILY_EXPLANATION_DICT, "family_formulas": FAMILY_FORMULA_DICT,
        "cards": cards_list, "findings": finding_list, "tests": test_dict, "pending": pending_list,
        "scope_count": len(STRATEGY_SPEC_LIST), "source_unchanged": True,
        "proof_limit": "ההשוואה למסלול החי מתחילה בכל מועד ממצב ייחוס. במסלולים החודשיים אומתו משקלי היעד ומועדי ההחלטה והביצוע; במסלולים היומיים נבדקו גם הוראות, מצב וכמויות לפי החוזה המותנה של המסלול. אין כאן הוכחה לחשבון עצמאי שמתקדם בין מילויים, או אישור ברוקר וסביבת פריסה.",
        "global_assumptions": ["הון המודל הוא מיליון דולר לכל אסטרטגיה בנפרד; התקציב החי הוא הון החשבון כפול אחוז התקציב, ואינו נקבע מעצם תווית WIRED.",
            "נתוני הספק הם הגרסה הנוכחית של ההיסטוריה. שחזור לפי קידומות אינו הוכחה לגרסת הנתונים שהייתה זמינה בעבר.",
            "תשואה שנתית מבוססת על זמן קלנדרי; תנודתיות ושארפ מחושבים מתשואות יומיות עם מאתיים חמישים ושניים ימי מסחר וריבית חסרת סיכון אפס.",
            "נתוני שנת הקורונה ושנת הריבית הם פירוק תיאורי קצר. שארפ של שנה בודדת אינו אומדן יציב, ואין לפרש אותו כהוכחה סטטיסטית.",
            "נתוני ההשוואה לבנצ׳מרק מיושרים לסגירה ראשונה משותפת. תוצאת האסטרטגיה לבדה כוללת גם את היום הראשון מול ההון הראשוני.",
            "מדדי הבנצ׳מרק כוללים דיבידנדים ברוטו. תשואת האסטרטגיה מנוכה מעלויות וניכוי המס שבמודל; אין טיפול מס ועמלות זהה בין שני הצדדים.",
            "עלויות הבסיס הן שתי נקודה חמש נקודות בסיס לכל צד, חצי סנט למניה ומינימום דולר; דיבידנדים בלונג בניכוי עשרים וחמישה אחוזים. ריבית מזומן חיובי אפס ומימון מזומן שלילי אינו מחושב במודל הבסיס.",
            "רגישויות העלות הן חיובים נוספים על יומן עסקאות קבוע. הן אינן הרצת אסטרטגיה מחדש ואינן כוללות תגובת פוזיציות, מימון או תשואה אבודה על העלות.",
            "מדדי המחזור הם מסחר יומי מצרפי, ולא עומק מכרז הפתיחה. היסטוריה מלאה ושלוש שנים אחרונות מוצגות בנפרד; נתוני מחזור חסרים אינם עוברים בדיקה.",
            "בדיקות נוספות שלא הורצו: שתים עשרה בדיקות מופעלות לפי בחירה, ובהן השוואות ישירות מול סנאפשוט ובדיקות HPI בהיסטוריה מלאה. השחזורים המדגמיים החדשים אינם מחליפים אותן; פירוט הסיבות נשמר בראיות.",
            "בדיקות מפעיל דורשות פירוט של מה שהורץ ומה שדולג. המתנה ב־Doctor אינה אישור של שערי ההמשך."],
        "post_deploy": ["לאמת את גרסת הקוד, קובץ ההפעלה, החשבון, סוג החשבון, אחוז התקציב והרשאות המסחר.",
            "לאמת פרופיל נתונים, תאריך, חתימות קבצים ותצפיות עזר; להשוות למצב החשבון ולמסד הפוזיציות הנכון.",
            "לאמת בעלות יחידה של המתזמן, מועד החלטה ושליחה, חגים, שעון קיץ וסגירה מוקדמת.",
            "להריץ Doctor ממוקד ולקרוא את הבדיקות שבוצעו; לטפל בשתי אי־ההתאמות היומיות שתועדו לפני הסתמכות עליהן.",
            "לאחר ביצוע מורשה לאסוף אישור קבלה, מילוי חתום, התאמה לחשבון וסיכום סוף יום; לבחון דחיות, ביצוע חלקי ועלויות בפועל."],
        "operator_improvements": ["עדיפות ראשונה: לבדוק זהות אחזקות בין בסיס ההחלטה לחשבון אצל שני ההורים הוותיקים, ולחסום פער גם בזמן הריצה. בדיקת Doctor מוקדמת לבדה אינה מונעת פער מאוחר.",
            "עדיפות ראשונה: להציג ולאמת תאריכי VIX, VXN ו־DTB3 בנפרד מתאריך הסנאפשוט. בכל סדרה נדרשת מדיניות גיל מפורשת שתואמת את הכלל המאושר.",
            "עדיפות שנייה: לתקן את אבחון מחזור הביצוע היומי ואת בדיקת הציטוט ב־CORE5. לשמור חסימה של הוראת פתיחה שלא נשלחה בזמן ושליחה לא ודאית ללא שידור חוזר אוטומטי.",
            "עדיפות שנייה: להציג כל רגל חניה בנפרד עם סכום ההוראה ומחזור קודם. מחזור חיובי או אחוזון קטן של כלל ההוראות עלולים להסתיר רגל SPMO גדולה.",
            "עדיפות שנייה: להוסיף לדשבורד שלוש חתימות נפרדות — מחקר, התאמת מימוש, וקבלת סביבת החשבון — עם גרסה, תאריך, תנאים וקישור לראיות. אין צורך במסך מסחר חדש.",
            "עדיפות שנייה: לעדכן הוראות מפעיל שהוחלפו. הפקודות בדשבורד יישארו להעתקה בלבד; בדיקות מימון מוקדמות שהוסרו בהחלטת הבעלים אינן מוחזרות במסגרת הביקורת."],
        "primary_sources": [{"title": "הגדרת נתוני פתיחה וסיווגים אצל Norgate", "url": "https://norgatedata.com/data-package-faq.php"},
            {"title": "עמלות מניות רשמיות אצל IBKR", "url": "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php"},
            {"title": "מגבלות סביבת הנייר של IBKR", "url": "https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/"}]}
    if owner_review_path.exists():
        report_dict = apply_owner_review(report_dict, read_json(owner_review_path))
    write_report_artifacts(report_dict)
    print(json.dumps({"complete": report_dict["complete"], "cards": len(cards_list), "findings": len(finding_list), "tests_passed": test_dict["passed"], "pending": pending_list}, ensure_ascii=False))


if __name__ == "__main__":
    main()
