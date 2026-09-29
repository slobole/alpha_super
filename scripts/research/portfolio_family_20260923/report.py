"""Build the Hebrew decision report from frozen portfolio result artifacts."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import markdown
import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923.analyze import metrics_dict
from scripts.research.portfolio_family_20260923.freeze import STUDY_PATH

CORE_LIST = ["CORE_100", "CORE_075", "CORE_050", "CORE_025", "CORE_000"]
NAME_DICT = {"CORE_100":"מאקרו הגנתי", "CORE_075":"הגנתי עם צמיחה", "CORE_050":"צמיחה מאוזנת",
             "CORE_025":"צמיחה מוגברת", "CORE_000":"צמיחה מלאה — תיק ייחוס"}
COLOR_LIST = ["#187c80", "#459cab", "#2c5aa0", "#b57c2b", "#8d4d74"]


def percent_str(value_float: float, digits_int: int = 1) -> str:
    return f"{value_float*100:.{digits_int}f}%" if pd.notna(value_float) else "לא זמין"


def table_str(header_list: list[str], row_list: list[list]) -> str:
    return "| " + " | ".join(header_list) + " |\n| " + " | ".join(["---"]*len(header_list)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str,row_list_obj)) + " |" for row_list_obj in row_list)


def evaluate_gates(metric_df: pd.DataFrame, period_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original gates without replacing any failed threshold."""
    row_list = []
    for scenario_str in ("common_account", "conservative"):
        main_df = metric_df.query("scenario == @scenario_str and rebalance == 'annual_fixed'").set_index("candidate_id")
        sub_df = period_df.query("scenario == @scenario_str and rebalance == 'annual_fixed'")
        for defensive_str, growth_str in zip(CORE_LIST[:-1], CORE_LIST[1:]):
            full_bool = all(main_df.loc[growth_str, field_str] >= main_df.loc[defensive_str,field_str] for field_str in ("volatility","es5_loss"))
            count_int = 0
            for period_int in (1,2,3):
                part_df = sub_df.query("period == @period_int").set_index("candidate_id")
                count_int += all(part_df.loc[growth_str,field_str] >= part_df.loc[defensive_str,field_str] for field_str in ("volatility","es5_loss"))
            row_list.append({"gate":"risk_order","scenario":scenario_str,"candidate":growth_str,"comparator":defensive_str,
                             "full_pass":full_bool,"partitions_passed":count_int,"pass":full_bool and count_int >= 2})
        for candidate_str in main_df.index:
            if candidate_str.startswith("DEF_"):
                comparator_str, gate_str = "CORE_100", "satellite"
                def passes_bool(candidate_series: pd.Series, reference_series: pd.Series) -> bool:
                    return bool(candidate_series.max_drawdown-reference_series.max_drawdown >= .01
                        and candidate_series.es5_loss <= reference_series.es5_loss*.95
                        and candidate_series.cagr >= reference_series.cagr-.005)
            elif candidate_str.startswith("L4_CORE5_"):
                comparator_str, gate_str = candidate_str.replace("CORE5","BIL"), "core_vs_bil"
                def passes_bool(candidate_series: pd.Series, reference_series: pd.Series) -> bool:
                    return bool(candidate_series.cagr >= reference_series.cagr+.005
                        and candidate_series.es5_loss <= reference_series.es5_loss*1.10
                        and candidate_series.max_drawdown >= reference_series.max_drawdown-.01)
            else:
                continue
            full_bool = passes_bool(main_df.loc[candidate_str],main_df.loc[comparator_str])
            count_int = 0
            for period_int in (1,2,3):
                part_df = sub_df.query("period == @period_int").set_index("candidate_id")
                count_int += passes_bool(part_df.loc[candidate_str],part_df.loc[comparator_str])
            row_list.append({"gate":gate_str,"scenario":scenario_str,"candidate":candidate_str,"comparator":comparator_str,
                             "full_pass":full_bool,"partitions_passed":count_int,"pass":full_bool and count_int >=2})
    return pd.DataFrame(row_list)


def build_charts(metric_df: pd.DataFrame, portfolio_return_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
    chart_path = STUDY_PATH/"charts"
    chart_path.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#24374c","text.color":"#24374c",
                         "axes.titleweight":"bold","figure.facecolor":"white","axes.grid":True,"grid.alpha":.16})
    anchor_ts = pd.Timestamp("2012-10-01")
    def normalized_series(return_series: pd.Series) -> pd.Series:
        return pd.concat([pd.Series([1.],index=[anchor_ts]),(1+return_series).cumprod()])
    figure_obj, axes_obj = plt.subplots(figsize=(11.5,4.8))
    for candidate_str, color_str in zip(CORE_LIST[:4],COLOR_LIST[:4]):
        nav_series = normalized_series(portfolio_return_df[candidate_str])
        axes_obj.plot(nav_series.index,nav_series*100,color=color_str,label=candidate_str.replace("CORE_","Macro ")+"%",lw=1.7)
    for benchmark_str,color_str in [("SPY_net25","#a8b0bd"),("BIL_net25","#444")]:
        nav_series = normalized_series(benchmark_df[benchmark_str])
        axes_obj.plot(nav_series.index,nav_series*100,color=color_str,label=benchmark_str,lw=1.2,ls="--")
    axes_obj.set(yscale="log",title="Growth of 100 | annual rebalance | common-account scenario",ylabel="Wealth index (log scale)")
    axes_obj.legend(ncol=3,frameon=False,fontsize=9)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"wealth.png",dpi=170); plt.close(figure_obj)
    figure_obj,axes_arr = plt.subplots(2,1,figsize=(11.5,6),sharex=True)
    for candidate_str,color_str in zip(CORE_LIST[:4],COLOR_LIST[:4]):
        nav_series = normalized_series(portfolio_return_df[candidate_str]); drawdown_series = nav_series/nav_series.cummax()-1
        axes_arr[0].plot(drawdown_series.index,drawdown_series,color=color_str,label=candidate_str,lw=1.1)
    spy_series = normalized_series(benchmark_df.SPY_net25)
    axes_arr[1].fill_between(spy_series.index,spy_series/spy_series.cummax()-1,0,color="#a8b0bd",alpha=.5,label="SPY_net25")
    for axes_obj in axes_arr:
        axes_obj.yaxis.set_major_formatter(PercentFormatter(1)); axes_obj.legend(ncol=4,frameon=False,fontsize=8)
    axes_arr[0].set_title("Drawdown from each path's previous peak | same dates, separate axes")
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"drawdown.png",dpi=170); plt.close(figure_obj)
    correlation_df = pd.read_csv(STUDY_PATH/"tables/rolling126_market_correlation.csv.gz",index_col=0,parse_dates=True)
    figure_obj,axes_obj = plt.subplots(figsize=(11.5,4))
    for candidate_str,color_str in zip(CORE_LIST[:4],COLOR_LIST[:4]):
        axes_obj.plot(correlation_df.index,correlation_df[candidate_str],label=candidate_str,color=color_str,lw=1)
    axes_obj.set(title="Correlation with SPY | trailing 126 sessions",ylim=(-1,1)); axes_obj.axhline(0,color="#777",lw=.6)
    axes_obj.legend(ncol=4,frameon=False); figure_obj.tight_layout(); figure_obj.savefig(chart_path/"rolling_correlation.png",dpi=170); plt.close(figure_obj)
    figure_obj,axes_obj = plt.subplots(figsize=(11.5,4))
    axis_vec = np.arange(5)
    for offset_float,scenario_str,label_str,color_str in [(-.18,"common_account","Common-account", "#287f95"),(.18,"conservative","Conservative costs","#d3a259")]:
        plot_df = metric_df.query("scenario==@scenario_str and rebalance=='annual_fixed'").set_index("candidate_id").loc[CORE_LIST]
        axes_obj.bar(axis_vec+offset_float,plot_df.cagr,width=.35,label=label_str,color=color_str)
    axes_obj.set_xticks(axis_vec,["Macro 100%","Macro 75%","Macro 50%","Macro 25%","Macro 0%"])
    axes_obj.set_title("Annualized return after costs | historical, not a forecast"); axes_obj.yaxis.set_major_formatter(PercentFormatter(1));axes_obj.legend(frameon=False)
    figure_obj.tight_layout();figure_obj.savefig(chart_path/"costs.png",dpi=170);plt.close(figure_obj)


def main() -> None:
    table_path = STUDY_PATH/"tables"
    metric_df = pd.read_csv(table_path/"portfolio_metrics.csv")
    period_df = pd.read_csv(table_path/"subperiod_metrics.csv")
    return_df = pd.read_csv(table_path/"primary_portfolio_returns.csv.gz",index_col=0,parse_dates=True)
    benchmark_df = pd.read_csv(table_path/"benchmark_returns.csv.gz",index_col=0,parse_dates=True)
    primary_df = metric_df.query("scenario=='common_account' and rebalance=='annual_fixed'").set_index("candidate_id")
    stress_df = metric_df.query("scenario=='conservative' and rebalance=='annual_fixed'").set_index("candidate_id")
    gate_df = evaluate_gates(metric_df,period_df);gate_df.to_csv(table_path/"frozen_gate_results.csv",index=False)
    benchmark_metric_df = pd.DataFrame([{"candidate_id":column_str,**metrics_dict(benchmark_df[column_str],benchmark_df.SPY_net25,benchmark_df.BIL_net25)} for column_str in benchmark_df]).set_index("candidate_id")
    benchmark_metric_df.to_csv(table_path/"benchmark_metrics.csv")
    native_row_list = []
    manifest_dict = json.loads((STUDY_PATH/"native_replay/replay_manifest.json").read_text(encoding="utf-8"))
    for case_dict in manifest_dict["case_metadata_list"]:
        nav_df = pd.read_csv(STUDY_PATH/"native_replay"/case_dict["case_id_str"]/"nav.csv.gz",index_col=0,parse_dates=True)
        native_return_series = nav_df.total_value.pct_change(fill_method=None).iloc[1:]
        native_row_list.append({"case_id":case_dict["case_id_str"],"capital":case_dict["capital_float"],**metrics_dict(native_return_series)})
    native_df = pd.DataFrame(native_row_list);native_df.to_csv(table_path/"native_replay_metrics.csv",index=False)
    build_charts(metric_df,return_df,benchmark_df)
    performance_rows = []
    for candidate_str in CORE_LIST:
        row_series = primary_df.loc[candidate_str]
        performance_rows.append([NAME_DICT[candidate_str],percent_str(row_series.cagr),percent_str(stress_df.loc[candidate_str,"cagr"]),percent_str(row_series.volatility),percent_str(row_series.max_drawdown),f"{row_series.sharpe_excess_bil:.2f}"])
    for candidate_str,label_str in [("SPY_net25","מניות ארה״ב — SPY"),("BIL_net25","אג״ח אוצר אמריקאי קצר — BIL")]:
        row_series = benchmark_metric_df.loc[candidate_str]
        performance_rows.append([label_str,percent_str(row_series.cagr),"—",percent_str(row_series.volatility),percent_str(row_series.max_drawdown),f"{row_series.sharpe_excess_bil:.2f}" if pd.notna(row_series.sharpe_excess_bil) else "—"])
    performance_table_str = table_str(["תיק","תשואה שנתית","בעלויות מחמירות","תנודתיות שנתית","ירידה מרבית","שארפ מעבר ל־BIL"],performance_rows)
    weight_table_str = table_str(["תיק","מאקרו","NDX","Mosaic","DV2","HPI"],[[NAME_DICT[candidate_str],f"{100-i_int*25}%"]+[f"{i_int*6.25:g}%"]*4 for i_int,candidate_str in enumerate(CORE_LIST)])
    monthly_rows = []
    for candidate_str in ["MONTHLY_TAA_100","MONTHLY_TAA_075","MONTHLY_TAA_050","MONTHLY_TAA_025","MONTHLY_TAA_000"]:
        taa_int=int(candidate_str[-3:]);row_series=primary_df.loc[candidate_str]
        monthly_rows.append([f"{taa_int}%",f"{(100-taa_int)/2:g}%",f"{(100-taa_int)/2:g}%",percent_str(row_series.cagr),percent_str(row_series.max_drawdown),percent_str(stress_df.loc[candidate_str,"cagr"])])
    monthly_table_str=table_str(["TAA ללא ETF מנייתי ממונף","NDX","Mosaic","תשואה שנתית","ירידה מרבית","תשואה בעלויות מחמירות"],monthly_rows)
    market_rows=[]
    for candidate_str in CORE_LIST[:4]:
        row_series=primary_df.loc[candidate_str]
        market_rows.append([NAME_DICT[candidate_str],f"{row_series.market_correlation:.2f}",f"{row_series.monthly_market_correlation:.2f}",f"{row_series.beta:.2f}",f"{row_series.annual_turnover:.1f}"])
    market_table_str=table_str(["תיק","מתאם יומי ל־SPY","מתאם חודשי","בטא","מחזור שנתי ביחס להון"],market_rows)
    report_str=rf'''# משפחת תיקי מאקרו וצמיחה

## סיכום והכרעה

**ההמלצה שלי היא לבנות משפחה פשוטה סביב ליבת המאקרו שלך: 100%, 75%, 50% ו־25% מאקרו.** את יתרת ההון מחלקים שווה בשווה בין שתי אסטרטגיות מומנטום ושתי אסטרטגיות קניית ירידות. כך הלקוח בוחר קודם כמה סיכון מנייתי הוא רוצה, ורק אחר כך מקבל הרכב. תיק ללא מאקרו נשאר נקודת ייחוס לצמיחה מלאה.

הבדיקה תומכת במדרגות האלה כמבנה עבודה: הפחתת המאקרו העלתה את התנודתיות ואת ההפסד הממוצע בימים הגרועים, בכל שלוש תקופות המשנה ובשתי הנחות העלות. **אלו תיקי מודל למחקר ולשיחת התאמה; אין כאן אישור למסחר בכספי לקוח או תחזית תשואה.** המשקולות נקבעו לפני התוצאות החדשות ולא נבחרו באופטימיזציה. ההיסטוריה כבר שימשה בפיתוח האסטרטגיות, ולכן תוצאה יפה אינה הוכחה עצמאית.

החלון המשותף הוא **2.10.2012–31.7.2026**, עם 3,476 תשואות יומיות, בדולרים. סכום הייחוס הוא מיליון דולר. המטבע טרם הוגדר על ידך, ולכן אין כאן בדיקה של שמירת ערך בשקלים.

## מה כל רכיב אמור לעשות

**מאקרו CORE5** עוקב אחר מגמות במניות ארה״ב, אג״ח ממשלת ארה״ב, זהב, סחורות ודולר. כל תחום מקבל 20%; תחום שאינו במגמה חיובית עובר ל־BIL. בנוסף יכול להיפתח שורט קטן בסחורות. ההגנה מגיעה מהפיזור ומהיציאה ממגמות חלשות; היא אינה ביטוח.

**מומנטום NDX ו־Mosaic** מחפש המשך עליות במניות חזקות. הראשון ממוקד בנאסד״ק ומקטין חשיפה לפי תנודתיות; השני פועל ביקום Russell 1000 ומשתמש גם בהפחתת דמיון בין אחזקות. **DV2 ו־HPI** קונים חולשה קצרת טווח לפי כללים שונים. הם מוסיפים דרך אחרת להרוויח בתוך מניות, אך עדיין עלולים להפסיד יחד עם המומנטום במשבר.

החלוקה השווה בתוך מנוע הצמיחה היא נקודת פתיחה שקופה. היא אינה טענה שאלו ארבע האסטרטגיות הטובות ביותר. כל 25 האסטרטגיות הזכאיות נסקרו, אבל ריבוי שמות דומים אינו בהכרח עוד פיזור. זוג מייצג מכל מנגנון שומר את ההרכב מובן ומקטין החלטות שניתן להתאים בדיעבד.

## ארבעה תיקים שאפשר להסביר

{weight_table_str}

המספרים הם אחוזים מההון, לא אחוזים מהסיכון. בתיק עם 75% מאקרו, המאקרו תרם כ־51% לשונות התיק; בתיק עם 50% מאקרו תרומתו הייתה רק כ־19%. כלומר, מחצית מהכסף באסטרטגיות צמיחה כבר סיפקה כ־81% מהשונות — מדד לעוצמת התנודות. זאת הסיבה שהייתי קורא לתיק הזה **צמיחה מאוזנת**, ולא ״חצי הגנתי״.

אין חובה להציע מיד ארבעה מוצרים. אפשר להתחיל בשלושה: מאקרו הגנתי, הגנתי עם צמיחה וצמיחה מאוזנת. התיק עם 25% מאקרו מתאים לדיון נפרד על נכונות לספוג ירידות. שם מסחרי צריך להיגזר מהסיכון שהלקוח מסוגל לשאת, ולא מהשארפ הגבוה ביותר בטבלה.

## מה קרה בהיסטוריה

{performance_table_str}

״תשואה שנתית״ היא קצב הצמיחה השנתי המורכב. ״ירידה מרבית״ היא הירידה הגדולה ביותר משיא קודם בתוך החלון, ולא גבול הפסד עתידי. שארפ מעבר ל־BIL מחושב על התשואה העודפת מול נכס מזומן דולרי באותם ימים; שארפ מול אפס מופיע בטבלאות המלאות.

ליבת המאקרו הניבה כאן כ־6.7% לשנה, עם ירידה מרבית של כ־5.3%. זה מתאים לתפקיד הגנתי ביחס למניות, אך **לא להצגה כקרן כספית**: במדגם הארוך שלה, מספטמבר 2007, הירידה הגיעה לכ־6.8% והתקופה הארוכה מתחת לשיא הייתה 764 ימי מסחר. היא מחזיקה נכסים מסוכנים וגם שורט; BIL לבדו הציג בחלון המשותף ירידה של כ־0.3% בלבד. התשואה ההיסטורית הנמוכה של BIL כוללת שנים של ריבית כמעט אפס ואינה אומדן לתשואה שלו כיום.

![צמיחת ההון](charts/wealth.png)

הגרף מתחיל ב־100 ומשתמש בסולם לוגריתמי. כל הקווים באותם תאריכים, אחרי הנחות המס והעלויות המתוארות בהמשך.

![ירידה משיא](charts/drawdown.png)

ל־SPY ציר נפרד, כדי שלא להסתיר את ההבדלים בין התיקים. בקטע משבר הקורונה שנקבע מראש, המאקרו עלה בכ־1.4%, בעוד תיק 50% מאקרו ירד בכ־8.3%. בקטע הירידות של 2022 המאקרו עלה בכ־3.2%, אך בתוכו ספג ירידה משיא של כ־5.3%. גם תשואה חיובית בתקופה יכולה לכלול דרך לא נעימה.

## האם להוסיף עוד אסטרטגיה הגנתית או TAA

**כרגע לא הייתי מוסיף אוטומטית Trinity, Crisis או VIXM לליבת המאקרו.** נבדקו 10% ו־20% מכל אחת. אף שילוב לא עבר את כל רף השיפור שנקבע מראש: ירידה מרבית טובה בלפחות נקודת אחוז, הפחתה של 5% בהפסד בימי קיצון, ואובדן תשואה שנתית שאינו עולה על חצי נקודת אחוז, באופן עקבי בתקופות ובעלויות. לדוגמה, 10% Crisis הורידו את הירידה המרבית לכ־4.4%, אבל התשואה ירדה לכ־5.9%: יש מחיר להגנה הזאת.

**TAA צומח הוא תוספת מעניינת, אך עדיין מועמד להמשך בדיקה.** בתיק 50% מאקרו, הקצאת 16.67% ל־TAA הצומח ו־8.33% לכל אחת מארבע אסטרטגיות הצמיחה העלתה את התשואה לכ־14.1% והקטינה את הירידה המרבית לכ־7.0%. מנגד, הוא יכול להחזיק TQQQ ממונף, והיתרון בתשואה לא היה מובהק לאחר תיקון לריבוי הבדיקות. אין הצדקה להחליף את ברירת המחדל הפשוטה רק משום שהגרף הזה נראה טוב יותר.

גם החלפת חלק מ־Ladder 4 במאקרו תרמה: 50% מאקרו והשאר משקולות Ladder 4 ביחס המקורי נתנו כ־14.3% לשנה וירידה של כ־7.1%. אבל מול החלפת אותו חלק ב־BIL, המאקרו הוסיף גם סיכון. לכן הוא **מנוע תשואה הגנתי**, ולא מזומן משופר ללא מחיר.

יש כאן ממצא ממוקד לטובת ההצעה שלך: החלפת 25% מ־Ladder 4 במאקרו עברה את סף התוספת שקבענו מול החלפה זהה ב־BIL, בשני תרחישי העלות. היא הניבה כ־18.0% לעומת 16.6% לשנה, עם ירידות מרביות דומות של כ־10.3% ו־10.1%. החלפות של 50% ו־75% לא עברו את אותו סף בגלל תוספת הסיכון. זה מועמד המשך מוגדר, ולא הצדקה להכריז שמאקרו תמיד עדיף על מזומן.

## מה לגבי 25% NDX ו־8% Mosaic

הבחירה שלך הגיונית כנטייה מכוונת לנאסד״ק, אבל מטריצת מתאמים לבדה אינה מצדיקה דווקא 25 מול 8. שתי האסטרטגיות חולקות מנגנון של מומנטום מנייתי. צריך לבדוק יחד מתאם בזמן ירידות, אחזקות משותפות, ריכוזיות ותרומה לסיכון.

בפועל המתאם ביניהן היה 0.72, ועלה ל־0.83 ב־5% מהימים הגרועים ביותר של SPY. החפיפה באותן מניות הייתה בממוצע כ־13% ממשקל כל אסטרטגיה, ובכל זאת הן נעו יחד. כלומר, מניות שונות אינן בהכרח סיכון שונה. ב־Ladder 4 תקציב המומנטום של 33% תרם כ־35% לשונות וכ־37% להפסד בימים הגרועים של התיק.

בבדיקה שמרה על תקציב מומנטום כולל של 33%, נבחנו גם חלוקות 0/33, 8/25, 16.5/16.5 ו־33/0 לצד 25/8 הקיים. הגדלת הנאסד״ק העלתה את התשואה בחלון המלא, אך גם את התנודתיות ואת ההפסד בימי קיצון. אף חלופה לא הוכיחה יתרון תשואה מתוקן לעומת 25/8. **לא מצאנו יחס מדויק שראוי לקרוא לו אופטימלי.** בתיקים החדשים אני מציע חצי־חצי בתוך תקציב המומנטום, כברירת מחדל ניתנת להסבר; הטיה נוספת לנאסד״ק צריכה להיות החלטת השקעה מודעת.

## חלופת הקרנות הענפיות

בדיקת המשך מוגבלת בחנה גם החלפת HPI בקניית ירידות בקרנות ענפיות. בתיק 50% מאקרו היא הקטינה את הירידה המרבית מ־9.0% ל־7.7%, אך הורידה את התשואה מ־13.5% ל־12.3%. הקצאת אותו חלק ל־BIL נתנה 11.5% וירידה של 7.0%. אף אחת משלוש ההחלפות לא עברה את כל ספי התועלת שנקבעו עבורה. זו חלופה מרככת עם מחיר, ולא שיפור חינם. הבדיקה הוגדרה לאחר תוצאות השלב הראשון, ולכן היא מחקר המשך על עבר מוכר.

## משפחה חודשית — מסלול נפרד

CORE5 בודק אותות מדי יום ויכול לסחור גם באמצע החודש. לכן תיק חודשי בלבד דורש ליבה אחרת. נבדקה ליבת TAA ללא מוצר מנייתי ממונף, עם אותות בסוף חודש וביצוע בפתיחה הבאה; אליה נוספו אותן שתי אסטרטגיות מומנטום חודשיות.

ההגדרה מתייחסת לפקודות של האסטרטגיות וללא מוצר כמו TQQQ. היא אינה מבטיחה שכל קרן מחזיקה רק מניות קנויות: BTAL עצמה מפעילה חשיפות קנייה ושורט בתוך הקרן.

{monthly_table_str}

המסלול חוסך מסחר, אך אינו מקביל לתיק המאקרו ההגנתי: גם 100% מה־TAA הזה ספגו ירידה של כ־11.1%. כמו כן, הוספת מניות דווקא הקטינה בחלק מהמקרים את הירידה המרבית ההיסטורית. לכן לא נכון להדביק למסלול הזה אותן תוויות סיכון רק על בסיס אחוז הליבה. 75% TAA ו־12.5% בכל מומנטום הם מועמד חודשי פשוט להמשך אימות, לא מנצח מוכח.

## הקשר לשוק ולעלויות

{market_table_str}

בטא מודדת רגישות לתנודות SPY; ערך נמוך אינו מבטיח הגנה ביום מסוים. מחזור המסחר הוא סך הקניות והמכירות חלקי ההון, ולכן מספר כמו 26 פירושו כ־26 פעמים ההון לשנה, ולא 26 עסקאות. הבדל זה מסביר מדוע תוספת עלות קטנה בכל צד פוגעת יותר בתיקים עם קניית ירידות.

![מתאם מתגלגל לשוק](charts/rolling_correlation.png)

![רגישות לעלויות](charts/costs.png)

תרחיש הבסיס משמר את עלויות האסטרטגיות, מאחד ניכוי של 25% מדיבידנד ארוך ומוסיף מימון היפותטי של 5% כשנדרש. בתרחיש המחמיר המימון הוא 8%, עלות השאלת שורט היא 5% במקום העלות המקורית, ונוספות 10 נקודות בסיס בכל צד מסחר. אלו מבחני רגישות, לא תעריפי ברוקר מוכחים. מס רווח הון, דמי ניהול, המרת מטבע והשפעת גודל פקודה לא נכללו.

הניכוי של 25% הוא הנחת חישוב על דיבידנד חיובי של אחזקה קנויה בלבד. הוא אינו מס של 25% על התשואה הכוללת ואינו קביעה לגבי שיעור המס של לקוח מסוים. המקדמים המקוריים לכל אסטרטגיה מופיעים בנספח.

## הכללים שמחזיקים את המבנה

בתחילת ההשקעה ובתחילת כל שנה מחזירים את משקל האסטרטגיות ליעד. בתוך השנה כל אסטרטגיה פועלת לפי כלליה והמשקולות צפות. לדוגמה, ללא איזון שנתי תנודתיות תיק 75% מאקרו עלתה כאן מ־6.4% לכ־8.6%. איזון הוא חלק ממדיניות ניהול הסיכון, לא פרט טכני.

האות המקורי מחושב ממידע זמין בסגירה, והמסחר בדרך כלל בפתיחה הבאה. איחוד התיקים הנוכחי נעשה באמצעות יחידות רעיוניות של כל אסטרטגיה; האיזון משתמש בהון של הסגירה הקודמת ומחיל את התשואה היומית הבאה. הוא עדיין אינו שחזור פקודות פיזי בפתיחה. נוכו 10 נקודות בסיס על ההקצאה הראשונה ועל כל שינוי מוחלט במשקל באיזון.

$$
r_{{p,t}}=(1-c_t)\sum_i w_{{i,t-1}}(1+r_{{i,t}})-1
$$

הנוסחה מתארת את תשואת התיק, עם משקולות שנקבעו לפני התשואה ועם עלות האיזון. מחוץ למועד איזון המשקולות משתנות לפי הביצועים; אין איפוס יומי נסתר.

$$
w'_{{i,t}}=\frac{{w_{{i,t-1}}(1+r_{{i,t}})}}{{\sum_j w_{{j,t-1}}(1+r_{{j,t}})}}
$$

בנוסף הורצה CORE5 במנוע האמיתי, מתחילת אוקטובר 2012, עם הון של 250 אלף עד מיליון דולר ומימון שמשפיע על הקניות הבאות. התשואה הייתה כ־6.68%–6.71% לשנה; בתרחיש המחמיר למיליון דולר כ־5.70%. הרצה ללא תוספת מימון התאימה בדיוק למנוע המקורי. זו בדיקת חשבונאות וגודל עבור CORE5; היא אינה מאשרת ביצוע בפועל של התיקים המשולבים.

רכיבי התיקים המשולבים מבוססים על ריצות מקור של 100 אלף דולר. התאמתן היחסית לסכום הייחוס אינה חישוב מחדש של מניות שלמות ועמלות מינימום בכל שרוול. בפרט, שרוול קטן של 62.5 אלף דולר מחייב בדיקת גודל נפרדת.

## מה הושלם ומה עדיין צריך להוכיח

נסקרו **25 אסטרטגיות: 9 WIRED ו־16 PM_READY**. נקבעו מראש 36 הרכבים, בשתי מדיניות איזון ובשלושה תרחישי עלות — 216 תאים; נוספו 75 בדיקות אסטרטגיה בודדת, שני מדדי ייחוס ושבע הרצות CORE5. אלה 300 תאי השלב הראשון. בדיקת הקרנות הענפיות הוסיפה שישה הרכבים ושלושה תרחישי עלות: **318 תאי חישוב בסך הכול**, לא הוכחות עצמאיות. 19 השוואות מקוריות נבדקו בדגימת בלוקים של 63 ימים ובתיקון Holm; שש השוואות ההמשך תוקנו בנפרד. שלוש השוואות המאקרו מול BIL הראו יתרון ממוצע מתוקן בחלון הנוכחי, אך התיקון אינו מוחק את מאות הניסיונות שנעשו בפיתוח הקודם.

נבדקו תאריכים משותפים ללא מילוי תשואות חסרות, עלויות, מס, שורט, תקופות משבר ותרומה לסיכון. חברות המדדים נשענות על מקורות Norgate ההיסטוריים. עדיין קיימים סיכונים של בחירת אסטרטגיות על היסטוריה מוכרת, ריצות ממועדי נתונים שונים וחוסר חותמת מלאה של הקוד שהיה בזמן יצירת כל ריצה. בחלק מאסטרטגיות המאקרו הנוספות הנתונים הם מגרסה עדכנית, ולא שחזור של הפרסום המקורי.

**הצעד הבא שמסוגל לשנות החלטה** הוא שחזור פקודות מלא להרכבים שנבחרו, באותו הון ובאותו מקור נתונים, ואז מעקב עתידי קפוא ללא שינוי משקולות לפי התוצאה. לפני התאמה ללקוח צריך להגדיר מטבע התחייבויות, צורך במשיכות, הפסד נסבל, אופק, מגבלות מינוף ושורט וגודל חשבון. כך בוחרים תיק לפי צורך; לא מבטיחים ללקוח את תשואת העבר.

ההכרעה: **לאמץ את המבנה כמשפחת מודלים למחקר ולשיחת התאמה; להשאיר את הליבה ההגנתית פשוטה; להימנע ממשקל ״אופטימלי״ מדומה; ולהמשיך לאימות ביצוע ולמדידה קדימה לפני הקצאת הון.** כל קובצי האסטרטגיות, התיקים והמסחר המקוריים נשארו ללא שינוי מטעם המחקר הזה.

[הנספח המלא](REPORT_FULL.md) · [כל 216 התוצאות](tables/portfolio_metrics.csv) · [25 האסטרטגיות](tables/all25_common_metrics.csv) · [כללי המחקר המקוריים](research_spec_frozen.json) · [בדיקות הסף](tables/frozen_gate_results.csv) · [מחברת הבדיקה](decision_notebook.ipynb)
'''
    state_dict=json.loads((STUDY_PATH/"research_state.json").read_text(encoding="utf-8"))
    report_str += ("\n\nסטטוס טכני למחקר: `research-only`; `verdict`: השערה להמשך; "
        "`replication`: התאמה מקומית של CORE5, ללא שחזור מלא של גרסאות המקור; "
        "`validation`: היסטוריה מוכרת בלבד; `runtime`: "
        + "תקציב יעד של 90 דקות ותקרה של 180 דקות; זמן העבודה בפועל מתועד ביומן המחקר.\n")
    (STUDY_PATH/"REPORT.md").write_text(report_str,encoding="utf-8")
    # The HTML shares one narrative with Markdown. Charts are inline base64 so
    # this primary report remains readable when copied as one standalone file.
    report_html_str=markdown.markdown(report_str,extensions=["tables","fenced_code"])
    import base64
    for image_path in (STUDY_PATH/"charts").glob("*.png"):
        report_html_str=report_html_str.replace(f'src="charts/{image_path.name}"',f'src="data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode()}"')
    # Display math remains readable without a remote script/font dependency.
    import re
    formula_list = ["Portfolio return = (1 − fee) × Σ [prior weight × (1 + sleeve return)] − 1",
                    "End weight = prior weight × (1 + sleeve return) / Σ [prior weight × (1 + sleeve return)]"]
    for formula_str in formula_list:
        report_html_str=re.sub(r"<p>\$\$.*?\$\$</p>",'<p class="formula">'+html.escape(formula_str)+'</p>',report_html_str,count=1,flags=re.S)
    full_table_html_str=metric_df.to_html(index=False,classes="all-results",float_format=lambda value_float:f"{value_float:.6g}")
    html_str='''<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>משפחת תיקי מאקרו וצמיחה | מחקר 23.09.2026</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#172f42;font-family:"Segoe UI",Arial,sans-serif;line-height:1.8;font-size:17px}header{background:#123848;color:white;padding:34px max(calc((100vw - 1080px)/2),28px)}header small{letter-spacing:.04em;color:#99c9d0}header strong{display:block;font-size:31px;line-height:1.4;margin:8px 0}header p{margin:0;color:#d5e5e7}.toolbar{position:sticky;top:0;background:#fffdf7f2;backdrop-filter:blur(12px);padding:10px 22px;border-bottom:1px solid #d6e2e5;z-index:2;display:flex;gap:18px;justify-content:center;align-items:center;font-size:14px}.toolbar a{color:#167784;text-decoration:none}button{border:1px solid #acc3c8;background:white;color:#154e58;padding:7px 13px;border-radius:7px;cursor:pointer}main{max-width:1120px;padding:32px 42px 64px;margin:24px auto;background:#fff;border:1px solid #dbe5e8;border-radius:16px;box-shadow:0 8px 30px #102e3d08}h1{display:none}h2{font-size:26px;line-height:1.4;margin:52px 0 18px;padding-top:12px;border-top:2px solid #e4edf0;color:#175866}h2:first-of-type{margin-top:0;border-top:0}p{margin:16px 0}strong{color:#103e50}table{width:100%;border-collapse:collapse;font-size:14px;margin:24px 0;line-height:1.55}th{background:#e8f1f3;color:#194a59;padding:12px 9px;text-align:right}td{padding:11px 9px;border-bottom:1px solid #dde7ea;font-variant-numeric:tabular-nums}td:not(:first-child){direction:ltr;unicode-bidi:isolate}tr:nth-child(even){background:#f8fafb}img{width:100%;height:auto;border:1px solid #e0e8eb;border-radius:8px;margin:9px 0}a{color:#087e87}code{direction:ltr;unicode-bidi:isolate;font-size:13px}details{border:1px solid #d5e2e7;padding:12px 18px;margin:26px 0;border-radius:10px}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto;direction:ltr;max-height:600px}.all-results{font-size:11px;white-space:nowrap}.formula{direction:ltr;text-align:center;overflow:auto;background:#f3f6f8;padding:16px;font-family:Consolas,monospace;font-size:14px}footer{color:#667e8b;font-size:13px;padding:16px;border-top:1px solid #ddd}@media(max-width:700px){body{font-size:16px}main{margin:12px;padding:22px 16px;border-radius:10px}header{padding:24px}header strong{font-size:25px}table{font-size:11px}td,th{padding:9px 4px}h2{font-size:23px}.toolbar{gap:12px;font-size:12px;flex-wrap:wrap}}@media print{body{background:white;font-size:11pt}.toolbar,details{display:none}main{margin:0;border:0;box-shadow:none;padding:12px}header{padding:20px}h2{break-after:avoid}table,img{break-inside:avoid}a{color:inherit}}
</style></head><body><header><small>ALPHA SUPER / מחקר תיקי מודל / 23.09.2026</small><strong>ליבה ברורה. מדרגות סיכון ברורות.</strong><p>ארבעה תיקי מאקרו וצמיחה, מסלול חודשי נפרד, והסבר למחיר של כל תוספת.</p></header><nav class="toolbar"><span>בדולרים · היסטוריה עד 31.07.2026 · מחקר בלבד</span><a href="REPORT_FULL.md">נספח מלא</a><a href="decision_notebook.ipynb">מחברת חישוב</a><button onclick="window.print()">הדפסה / שמירה כ־PDF</button></nav><main>'''+report_html_str+'''<details><summary>כל 216 תאי ההשוואה — נתונים גולמיים</summary><p>השדות המספריים הם שברים עשרוניים. לדוגמה 0.10 = 10%. אפשר להוריד את קובץ ה־CSV דרך הקישור בדוח.</p><div class="scroll">'''+full_table_html_str+'''</div></details><footer>המסמך מציג תוצאות מחקר היסטוריות. כל המשקולות מוגדרות בתוכנית הקפואה; שום הגדרת מסחר חיה לא שונתה.</footer></main></body></html>'''
    html_str=html_str.replace('header strong{display:block;','header strong{color:white;display:block;').replace('href="REPORT_FULL.md"','href="REPORT_FULL.html"')
    (STUDY_PATH/"REPORT.html").write_text(html_str,encoding="utf-8")
    print(json.dumps({"report":"REPORT.html","gates":gate_df.groupby("gate")["pass"].agg(["sum","count"]).to_dict(),"native_replays":len(native_df)},ensure_ascii=False))


if __name__ == "__main__":
    main()
