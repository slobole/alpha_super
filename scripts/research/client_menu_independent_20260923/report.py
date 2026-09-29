"""Produce a readable Hebrew menu and full, source-linked research appendix."""
from __future__ import annotations
from datetime import datetime, timezone
import base64
import html
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import markdown
from scripts.research.client_menu_independent_20260923.protocol import ROOT_PATH, SOURCE_PATH, STUDY_PATH, digest_str, write_json
from scripts.research.client_menu_independent_20260923.analyze import horizon_metrics
from scripts.research.portfolio_family_20260923 import analyze as ledger

LABEL_DICT = {"S2": "Macro stability", "M0": "Monthly growth", "A3": "Active sector growth", "G2": "Aggressive growth", "SPY": "SPY", "BIL": "BIL", "Ladder4": "Existing Ladder4"}
HEBREW_DICT = {"S2": "מאקרו מתון", "M0": "צמיחה חודשית", "A3": "צמיחה עם פעילות יומית", "G2": "צמיחה אגרסיבית", "SPY": "SPY", "BIL": "BIL", "Ladder4": "Ladder 4 הקיים"}
COLOR_DICT = {"S2": "#287a73", "M0": "#3766b1", "A3": "#8f66a5", "G2": "#d77b32", "SPY": "#697687", "BIL": "#acb4bc", "Ladder4": "#98816b"}


def pct(value_float: float, digits_int: int = 1) -> str:
    return f"{value_float*100:.{digits_int}f}%"


def comparison() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec_dict = json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    for input_dict in spec_dict["input_files"]:
        ledger.verify_file(input_dict)
    ledger.preflight_source_ids(spec_dict, json.loads((SOURCE_PATH/"source_audit_addendum.json").read_text()))
    if not json.loads((STUDY_PATH/"active_alternative_verdict.json").read_text())["accepted_provisionally"]:
        raise ValueError("A3 did not qualify for provisional presentation")
    if json.loads((STUDY_PATH/"selection.json").read_text())["policy_priority_selection"] != {"S": "S2", "M": "M0", "A": None, "G": "G2"}:
        raise ValueError("Report narrative must be reviewed after a selection change")
    active_dict = json.loads((STUDY_PATH/"adaptive_active_spec.json").read_text())
    comparison_spec_path = STUDY_PATH/"presentation_comparison_spec.json"
    if not comparison_spec_path.exists():
        write_json(comparison_spec_path, {"frozen_at": datetime.now(timezone.utc).isoformat(), "purpose": "Common-date menu comparison after A3follow-up; no selection or weight changes", "anchor": active_dict["anchor"], "end": active_dict["end"], "new_cells": 6, "candidates": ["S2", "G2", "Ladder4"], "scenarios": ["common_account", "conservative"], "source_results_sha256": digest_str(STUDY_PATH/"tables/results.csv"), "active_results_sha256": digest_str(STUDY_PATH/"tables/active_alternative.csv")})
    comparison_spec_dict = json.loads(comparison_spec_path.read_text())
    for hash_key_str, result_name_str in [("source_results_sha256", "results.csv"), ("active_results_sha256", "active_alternative.csv")]:
        if comparison_spec_dict[hash_key_str] != digest_str(STUDY_PATH/"tables"/result_name_str):
            raise ValueError(f"Presentation input changed: {result_name_str}")
    catalog_list = json.loads((SOURCE_PATH/"catalog_complete.json").read_text(encoding="utf-8"))
    alias_id_dict = {row_dict["alias"]: row_dict["strategy_import"].split(":")[0].split(".")[-1] for row_dict in catalog_list}
    alias_id_dict.update({"BIL": "benchmark_bil", "SPY": "benchmark_spy"})
    wanted_list = ["TRINITY", "CORE5", "MOSAIC", "DF_BTAL_QQQ_LINEAR", "SECTOR_6", "HPI", "DF_BTAL_TQQQ_EQUAL", "DV2", "HPI_VOTE", "NDX_VXN", "DF_BTAL_TQQQ_RANK", "BIL", "SPY"]
    component_dict = {alias_str: ledger.source_components(SOURCE_PATH/("benchmarks" if alias_str in {"BIL", "SPY"} else "data")/alias_id_dict[alias_str])[0] for alias_str in wanted_list}
    weight_dict = {candidate_str: spec_dict["portfolio"]["candidates"][candidate_str]["weights"] for candidate_str in ["S2", "M0", "G2"]}
    weight_dict["A3"] = active_dict["weights"]
    weight_dict["Ladder4"] = {"DV2": .16, "HPI_VOTE": .17, "NDX_VXN": .25, "MOSAIC": .08, "DF_BTAL_TQQQ_RANK": .34}
    active_df = pd.read_csv(STUDY_PATH/"tables/active_alternative.csv")
    result_list = []
    for candidate_str, stored_str in [("M0", "M0_matched"), ("A3", "A3"), ("BIL", "BIL_matched"), ("SPY", "SPY_matched")]:
        for _, row_series in active_df.loc[(active_df.candidate == stored_str)&active_df.scenario.isin(["common_account", "conservative"])].iterrows():
            result_list.append({**row_series.to_dict(), "candidate": candidate_str})
    return_dict = {}
    for candidate_str in ["S2", "G2", "Ladder4"]:
        for scenario_str in ["common_account", "conservative"]:
            panel_df = ledger.strict_return_panel(component_dict, list(dict.fromkeys([*weight_dict[candidate_str], "SPY", "BIL"])), active_dict["anchor"], active_dict["end"], scenario_str)
            nav_series, start_weight_df, contribution_df, cost_series = ledger.simulate_book(panel_df, weight_dict[candidate_str], pd.Timestamp(active_dict["anchor"]), "annual_fixed", .001)
            return_series = nav_series.pct_change(fill_method=None).iloc[1:]
            result_list.append({"candidate": candidate_str, "scenario": scenario_str, **ledger.metrics_dict(return_series, panel_df.SPY, panel_df.BIL), **horizon_metrics(return_series)})
            if scenario_str == "common_account":
                return_dict[candidate_str] = return_series
    active_return_df = pd.read_csv(STUDY_PATH/"data/active_alternative_returns.csv.gz", index_col=0, parse_dates=True).rename(columns={"M0_matched": "M0"})
    for candidate_str in active_return_df:
        return_dict[candidate_str] = active_return_df[candidate_str]
    base_panel_df = ledger.strict_return_panel(component_dict, ["SPY", "BIL"], active_dict["anchor"], active_dict["end"], "common_account")
    for candidate_str in ["SPY", "BIL"]:
        # Same exact initial allocation fee as stored matched benchmark metrics.
        return_series = base_panel_df[candidate_str].copy()
        return_series.iloc[0] = (1+return_series.iloc[0])*.999-1
        return_dict[candidate_str] = return_series
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(STUDY_PATH/"tables/menu_common_sample.csv", index=False)
    menu_return_df = pd.DataFrame(return_dict)
    if menu_return_df.isna().any().any():
        raise ValueError("Comparison calendars differ")
    menu_return_df.to_csv(STUDY_PATH/"data/menu_common_returns.csv.gz")
    # A3's separate operating-cost and exposure diagnostics on its actual dates.
    panel_df = ledger.strict_return_panel(component_dict, list(weight_dict["A3"]), active_dict["anchor"], active_dict["end"], "common_account")
    _, start_weight_df, _, _ = ledger.simulate_book(panel_df, weight_dict["A3"], pd.Timestamp(active_dict["anchor"]), "annual_fixed", .001)
    eod_weight_df = start_weight_df.mul(1+panel_df)
    eod_weight_df = eod_weight_df.div(eod_weight_df.sum(axis=1), axis=0)
    asset_book_df = pd.DataFrame(index=panel_df.index)
    turnover_series = pd.Series(0., index=panel_df.index)
    for alias_str in weight_dict["A3"]:
        holding_df = ledger.read_frame(SOURCE_PATH/"data"/alias_id_dict[alias_str]/"realized_weights.csv.gz", True)
        # *** CRITICAL *** Sparse unheld asset cells, not absent dates, become0.
        holding_df = holding_df.loc[panel_df.index].drop(columns=["Cash"]).fillna(0.)
        asset_book_df = asset_book_df.add(holding_df.mul(eod_weight_df[alias_str], axis=0), fill_value=0.)
        turnover_series += start_weight_df[alias_str]*component_dict[alias_str].turnover.loc[panel_df.index]
    shock_dict = json.loads((STUDY_PATH/"amendment_002_fragility.json").read_text())["shocks"]["liquidation"]
    shock_series = asset_book_df.mul(pd.Series({asset_str: shock_dict.get(asset_str, shock_dict["equity"]) for asset_str in asset_book_df})).sum(axis=1)
    active_exposure_dict = {"annual_oneway_turnover": float(turnover_series.mean()*252), "max_gross": float(asset_book_df.abs().sum(axis=1).max()), "worst_liquidation_state": float(shock_series.min()), "scope": "post-result descriptive active-book native-holdings blend; scenario and dates disclosed"}
    write_json(STUDY_PATH/"active_exposures.json", active_exposure_dict)
    return result_df, menu_return_df, weight_dict


def charts(menu_return_df: pd.DataFrame) -> None:
    chart_path = STUDY_PATH/"charts"
    chart_path.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    candidate_list = ["S2", "M0", "A3", "G2", "SPY", "BIL"]
    anchor_ts = pd.Timestamp("2018-07-19")
    nav_df = pd.concat([pd.DataFrame(1., index=[anchor_ts], columns=candidate_list), (1+menu_return_df[candidate_list]).cumprod()])
    figure_obj, axes_obj = plt.subplots(figsize=(11.5, 5))
    for candidate_str in candidate_list:
        axes_obj.plot(nav_df.index, nav_df[candidate_str]*100, label=LABEL_DICT[candidate_str], color=COLOR_DICT[candidate_str], linewidth=2 if candidate_str not in {"SPY", "BIL"} else 1.2)
    axes_obj.set_yscale("log"); axes_obj.set_ylabel("Growth of 100 USD, log scale")
    axes_obj.set_title("Identical sample: 20 Jul 2018 – 31 Jul 2026 | central costs | annual sleeve reset")
    axes_obj.grid(alpha=.2); axes_obj.legend(ncol=3, frameon=False)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"equity.png", dpi=160); plt.close(figure_obj)
    figure_obj, axes_obj = plt.subplots(figsize=(11.5, 4.5))
    for candidate_str in candidate_list[:-1]:
        drawdown_series = nav_df[candidate_str]/nav_df[candidate_str].cummax()-1
        axes_obj.plot(drawdown_series.index, drawdown_series*100, color=COLOR_DICT[candidate_str], label=LABEL_DICT[candidate_str], linewidth=1.4)
    axes_obj.set_ylabel("Drawdown (%)"); axes_obj.set_title("Realized historical loss is not a forward loss limit")
    axes_obj.legend(ncol=3, frameon=False); axes_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"drawdown.png", dpi=160); plt.close(figure_obj)
    annual_df = (1+menu_return_df[candidate_list]).resample("YE").prod()-1
    figure_obj, axes_obj = plt.subplots(figsize=(11.5, 4.5))
    image_obj = axes_obj.imshow(annual_df.T.to_numpy()*100, aspect="auto", cmap="RdYlGn", vmin=-25, vmax=35)
    axes_obj.set_yticks(range(len(candidate_list)), [LABEL_DICT[candidate_str] for candidate_str in candidate_list])
    axes_obj.set_xticks(range(len(annual_df)), [str(date_ts.year)+("*" if date_ts.year in {2018, 2026} else "") for date_ts in annual_df.index])
    for row_int in range(len(candidate_list)):
        for column_int in range(len(annual_df)):
            axes_obj.text(column_int, row_int, f"{annual_df.iloc[column_int,row_int]*100:.1f}%", ha="center", va="center", fontsize=9)
    axes_obj.set_title("Calendar-year returns | *2018/2026 are partial years | central costs")
    figure_obj.colorbar(image_obj, ax=axes_obj, label="Return (%)", fraction=.025)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"annual.png", dpi=160); plt.close(figure_obj)
    figure_obj, axes_obj = plt.subplots(figsize=(11.5, 4.5))
    for candidate_str in candidate_list[:4]:
        # *** CRITICAL *** Trailing126-day reporting statistic; no portfolio input.
        correlation_series = menu_return_df[candidate_str].rolling(126, min_periods=126).corr(menu_return_df.SPY)
        axes_obj.plot(correlation_series.index, correlation_series, color=COLOR_DICT[candidate_str], label=LABEL_DICT[candidate_str], linewidth=1.5)
    axes_obj.set_ylim(-.5, 1.); axes_obj.set_ylabel("Trailing126-day correlation with SPY")
    axes_obj.set_title("Market dependence changes over time; four names do not mean four independent risks")
    axes_obj.legend(ncol=2, frameon=False); axes_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path/"rolling_correlation.png", dpi=160); plt.close(figure_obj)


def render_html(markdown_str: str, title_str: str) -> str:
    markdown_str = re.sub(r"\$\$(.*?)\$\$", lambda match_obj: '<pre class="formula">'+html.escape(match_obj.group(1).strip())+'</pre>', markdown_str, flags=re.S)
    body_str = markdown.markdown(markdown_str, extensions=["tables", "fenced_code"])
    body_str = re.sub(r'(<td\b[^>]*>)(-?\d+(?:\.\d+)?%?)(</td>)', r'\1<bdi dir="ltr">\2</bdi>\3', body_str)
    body_str = body_str.replace('50% Trinity + 50% CORE5', '<bdi dir="ltr">50% Trinity + 50% CORE5</bdi>')
    for image_path in (STUDY_PATH/"charts").glob("*.png"):
        body_str = body_str.replace(f'src="charts/{image_path.name}"', f'src="data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode()}"')
    return f'''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title_str)}</title><style>
    :root{{--ink:#183039;--muted:#586a73;--line:#d9e2e4;--accent:#22786e}}*{{box-sizing:border-box}}body{{margin:0;background:#edf2f1;color:var(--ink);font:18px/1.75 "Segoe UI",Arial,sans-serif}}main{{max-width:1120px;margin:30px auto;background:white;padding:46px 54px;border-top:7px solid var(--accent);box-shadow:0 10px 40px #233c3810}}h1{{font-size:40px;line-height:1.2;margin-top:8px}}h2{{font-size:27px;border-top:1px solid var(--line);padding-top:25px;margin-top:40px}}h3{{font-size:22px}}p{{margin:14px 0}}a{{color:#246a9c;text-decoration:none}}strong{{font-weight:650}}table{{width:100%;border-collapse:collapse;font-size:15px;display:block;overflow-x:auto;white-space:normal}}th{{background:#eaf1f0;text-align:right}}td,th{{padding:10px 12px;border-bottom:1px solid var(--line)}}td{{vertical-align:top}}img{{max-width:100%;height:auto;display:block;margin:18px auto}}pre{{direction:ltr;text-align:left;background:#f4f6f7;padding:16px;overflow:auto;font-size:13px}}code{{direction:ltr;unicode-bidi:embed;overflow-wrap:anywhere;font-size:.84em}}blockquote{{margin:20px 0;padding:14px 22px;background:#eef6f3;border-right:4px solid var(--accent)}}.kicker{{font-size:14px;color:var(--muted);letter-spacing:.04em}}.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.card{{border:1px solid var(--line);border-top:4px solid var(--accent);padding:20px;background:#fbfdfc}}.card h3{{margin:0 0 9px}}.card p{{margin:8px 0;font-size:16px}}.badge{{font-size:12px;color:#80601d;background:#fcf2d9;padding:3px 8px;border-radius:4px}}details{{border-bottom:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:600}}.source{{font-size:14px;color:var(--muted)}}@media(max-width:700px){{body{{font-size:16px}}main{{margin:0;padding:25px 18px}}h1{{font-size:30px}}h2{{font-size:24px}}.cards{{grid-template-columns:1fr}}td,th{{padding:8px}}}}@media print{{body{{background:white;font-size:12px}}main{{box-shadow:none;margin:0;padding:12px}}h2{{break-after:avoid}}.card,figure{{break-inside:avoid}}table{{display:table;font-size:10px}}details{{display:block}}}}</style><main>{body_str}</main></html>'''


def main() -> None:
    comparison_df, return_df, weight_dict = comparison()
    charts(return_df)
    primary_df = pd.read_csv(STUDY_PATH/"tables/results.csv")
    central_df = comparison_df.loc[comparison_df.scenario == "common_account"].set_index("candidate")
    conservative_df = comparison_df.loc[comparison_df.scenario == "conservative"].set_index("candidate")
    main_table_str = "| תיק | תשואה שנתית היסטורית | תנודתיות | Sharpe¹ | ירידה מרבית | שנה מתגלגלת גרועה | בטא למניות |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for candidate_str in ["S2", "M0", "A3", "G2", "SPY", "BIL", "Ladder4"]:
        row_series = central_df.loc[candidate_str]
        main_table_str += f"| {HEBREW_DICT[candidate_str]} | {pct(row_series.cagr)} | {pct(row_series.volatility)} | {row_series.sharpe_zero:.2f} | {pct(row_series.max_drawdown)} | {pct(row_series.worst_rolling252)} | {row_series.beta:.2f} |\n"
    cost_table_str = "| תיק | תרחיש מרכזי | תרחיש עלויות מחמיר |\n|---|---:|---:|\n"
    for candidate_str in ["S2", "M0", "A3", "G2"]:
        cost_table_str += f"| {HEBREW_DICT[candidate_str]} | {pct(central_df.loc[candidate_str,'cagr'])} | {pct(conservative_df.loc[candidate_str,'cagr'])} |\n"
    active_exposure_dict = json.loads((STUDY_PATH/"active_exposures.json").read_text())
    narrative_str = f'''<div class="kicker">ALPHA SUPER · מחקר תכנון תיקים · 23 בספטמבר 2026</div>

# תפריט קטן. לכל תיק תפקיד ברור.

**סיכום:** אני מציע שלושה מועמדים עיקריים — מאקרו מתון, צמיחה חודשית וצמיחה אגרסיבית — ולצדם מועמד פעיל עם היסטוריה קצרה יותר. כל אחד נבנה לצורך אחר מתוך כלל 25 האסטרטגיות הזמינות. אין כרגע הצדקה לתאר אחד מהם כ״קרן כספית טובה יותר״. אלו הצעות מחקר לתכנון מוצר; הבדיקה אינה אישור להקצות כסף או הבטחת תשואה.

המנדט קדם לתוצאות: הוגדרו תקרות הפסד היסטוריות לבחינה, אופקי החזקה, מגבלות מסחר וסיבה להוסיף כל מנגנון. אחר כך נבדקו עלויות, חלופות, הסרת רכיבים ושינוי משקולות. **CORE5 מופיע רק במוצר המאקרו.** ההרכבים אינם פתרון של מקסום Sharpe.

## ארבעה שימושים, ארבעה הרכבים

<div class="cards">
<article class="card"><h3>1 · מאקרו מתון</h3><p><strong>50% Trinity + 50% CORE5</strong></p><p>ללקוח שמעדיף תלות נמוכה יותר בשוק המניות ומוכן לצמיחה מתונה ולתקופות הפסד. אופק תכנוני: שלוש שנים ומעלה; מסחר בתדירות מעורבת.</p><p>Trinity משנה את החשיפה למניות, זהב ואג״ח ארוכות לפי תנודתיות. CORE5 עוקב אחר מגמות בחמישה שווקים ועובר לשטרי אוצר כשהמגמה נחלשת. השילוב מחבר בקרת גודל חשיפה עם בחירת כיוון.</p><p><strong>דורש יכולת לשורט קטן ב־DBC.</strong> שני המנגנונים חולקים חשיפה לזהב ולאג״ח, ולכן אינם שתי הגנות בלתי תלויות.</p></article>
<article class="card"><h3>2 · צמיחה חודשית</h3><p><strong>50% MOSAIC + 50% Defense First לינארי עם QQQ</strong></p><p>ללקוח שרוצה צמיחה ומחזור החלטות מתוכנן אחד בחודש. אופק תכנוני: חמש שנים ומעלה.</p><p>MOSAIC בוחר 20 מניות מומנטום מתוך Russell 1000 ומעניש בחירה במניות דומות מדי. הרכיב הטקטי מחלק בין זהב, דולר, אג״ח, סחורות ו־BTAL; חלקים שאינם עוברים את מסנן המגמה יכולים לעבור ל־QQQ או למזומן.</p><p>50/50 נותן תקציב שווה לשני מנגנונים. אין ETF מנייתי ממונף, אך BTAL מכיל פעילות לונג־שורט פנימית.</p></article>
<article class="card"><h3>3 · צמיחה עם פעילות יומית</h3><p><strong>⅓ MOSAIC + ⅓ חזרה לממוצע ב־6 קרנות סקטוריאליות + ⅓ Defense First לינארי עם QQQ</strong></p><p>ללקוח שמוכן לפעילות יומית כדי למתן ימי הפסד, גם במחיר עלויות גבוהות יותר. אופק תכנוני: חמש שנים ומעלה.</p><p>קרנות הסקטור הן SOXX, IGV, IBB, KIE, IHI ו־XLC. קניית חולשה קצרת טווח מצטרפת למומנטום ולהקצאה טקטית. זו עדיין חשיפה מנייתית, לא גידור.</p><p><span class="badge">מועמד משני: היסטוריה רק מיולי 2018</span></p><p>המודל הסקטוריאלי משתמש בשברי מניות. התאמה לכמויות ולאופן ביצוע אמיתי נותרה תנאי לפני הקצאה.</p></article>
<article class="card"><h3>4 · צמיחה אגרסיבית</h3><p><strong>⅓ MOSAIC + ⅓ HPI בסיסי + ⅓ Defense First במשקל שווה עם TQQQ</strong></p><p>ללקוח בעל אופק של שבע שנים ומעלה, נכונות לירידות חדות ותפעול יומי.</p><p>MOSAIC מחפש המשכיות, HPI קונה ירידה חריגה בתוך מגמה ארוכה חיובית, והרכיב הטקטי יכול להפנות הון ל־TQQQ. המינוף הוא חלק מהסיכון של המוצר, ולא פרט טכני.</p><p><strong>זעזוע מניות היפותטי של 20% יצר הפסד תיק של עד כ־35.2% במצבי אחזקה היסטוריים.</strong> זו המחשה, לא תחזית ולא תקרה.</p></article>
</div>

שמות האסטרטגיות המלאים והמשקולות המדויקות מופיעים ב[נספח](REPORT_FULL.html). כל התיקים נקובים בדולר. לקוח שמודד התחייבויות בשקלים חשוף גם לשינוי בשער החליפין; המחקר אינו כולל גידור מטבע.

## למה דווקא המשקולות האלה

נקודת המוצא היא **תקציב הון שווה לכל מנגנון שנבחר**. שני מנגנונים מקבלים חצי־חצי; שלושה מקבלים שליש־שליש. זהו כלל תכנון פשוט שאפשר להסביר ולעדכן, לא טענה שכל מנגנון תורם סיכון שווה. כל רכיב צריך להוכיח שימושיות בתוך התיק — תשואה לבדה אינה מספיקה להגנה, וקורלציה ממוצעת נמוכה אינה מספיקה לפיזור.

נבדקו העברות של עשר נקודות אחוז בין הרכיבים, כולל שינוי משקל הרכיב הממונף. המועמדים נשארו בתוך מבחני ההפסד והאופק שנקבעו. בדיקות המשקולות אינן מוכיחות ש־50% טוב מ־45%; הן מקטינות את החשש שהמסקנה תלויה במספר מדויק. חלק מההרחבה נעשה אחרי התוצאות ומתועד כך.

משקולות אלה הן יעדי הקצאה בתחילת התקופה ובתחילת כל שנה. בתוך השנה כל POD צובר תשואה בנפרד והמשקל שלו משתנה. אין איזון יומי סמוי. בתיק האגרסיבי משקל TQQQ בפועל הגיע לכ־39.8% מהתיק; לכן שליש אינו תקרת חשיפה רציפה.

שווי התיק הוא סכום חשבונות האסטרטגיות; הירידה נמדדת מול השווי הגבוה ביותר שנרשם עד אותו יום:

$$
E(portfolio,t) = Σ E(POD_i,t)
$$

$$
DD_t = E_t / max(E_0, ..., E_t) - 1
$$

## השוואה על אותם תאריכים

כדי לכלול את המועמד הסקטוריאלי ביושר, הטבלה והגרפים כאן משתמשים ב־**20.7.2018–31.7.2026**, עם עוגן שווי ב־19.7.2018. ההרכבים נקבעו מחדש בתחילת חלון ההשוואה. כל המספרים הם היסטוריים, לאחר עלויות המקור והתאמות החשבון המרכזיות, לפני דמי ניהול ומס רווח הון של לקוח. הם אינם אומדן לתשואה העתידית.

{main_table_str}

¹ Sharpe מחושב כאן בריבית חסרת סיכון אפס. התנודתיות הזעירה של BIL אינה הופכת יחס גבוה להוכחת עדיפות; בטבלאות המלאות נשמר גם יחס עודף מול BIL.

לשלושת המועמדים הראשיים יש גם בדיקה ארוכה יותר, מאוקטובר 2012: מאקרו מתון הניב 6.2% לשנה עם ירידה מרבית של 10.0%; החודשי 15.5% עם 9.0%; האגרסיבי 21.6% עם 14.7%. לתיק המאקרו נבדקה גם ההיסטוריה האמיתית מ־2008, ללא יצירת היסטוריה מלאכותית. התיק החודשי והאגרסיבי כוללים BTAL ולכן אין להם תיק מלא שנבדק ב־2008.

![שווי התיקים](charts/equity.png)

![ירידות מהשיא](charts/drawdown.png)

המאקרו אינו ״הטוב ביותר על הגרף״: בתקופה הארוכה החודשי סיפק תשואה גבוהה יותר ואף ירידה מרבית מעט קטנה יותר. הסיבה להשאיר מוצר מאקרו היא התלות הנמוכה יותר במניות ומנגנון החשיפה האחר — לא ניצחון היסטורי בכל מדד. בתקופה הארוכה הבטא שלו הייתה כ־0.15, מול כ־0.37 בחודשי וכ־0.66 באגרסיבי.

## מה תרמה העבודה מעבר לבחירת תיקים

**אין סיבה אוטומטית להוסיף חזרה לממוצע.** שליש HPI לצד MOSAIC והקצאה טקטית לא שיפר מספיק את ההפסדים מול התיק החודשי. דווקא גרסת שש הקרנות הסקטוריאליות נתנה בחלון שלה תשואה שנתית דומה — 15.2% מול 15.1% — והפסד ממוצע ב־5% הימים הגרועים קטן בכ־14%. בתרחיש העלויות המחמיר היא נחלשה יותר מהחודשי. היא נכנסה כמועמד משני, עם היסטוריה קצרה ושברי מניות גלויים, בעקבות בדיקת המשך מתועדת.

**NDX ו־MOSAIC אינם שני פקטורים עצמאיים.** הם שני מימושים של מומנטום מניות. העדפתי MOSAIC בתיק החודשי כבסיס רחב יותר, ולא משום שאחוז מסוים שלו נמצא ״אופטימלי״. החלפתו ב־NDX עם VXN העלתה בתצפית הארוכה את התשואה מ־15.5% ל־17.6%, אך גם את הירידה המרבית מ־9.0% ל־11.8%. פיצול ל־25% MOSAIC, ל־25% NDX ול־50% הקצאה טקטית נתן 16.6% ו־9.0%. זו חלופה סבירה למי שמבקש יותר הטיית נאסדק; אין בסיס סטטיסטי נקי לקבוע שיחס 25/8 הוא האמת הנכונה לכל לקוח.

**תווית ״הגנה״ לא הספיקה.** Trinity לבדו רשם ירידה של 17.1% ושנה מתגלגלת שלילית של 15.8%, ונפסל למנדט המתון שהוגדר. שילובים עם VIXM או Crisis Trend לא הצדיקו אוטומטית הקצאה קבועה מול בקרת מזומן. Tactical Fixed Income שיפר חלק ממדדי הסיכון, אך גרסת חצי־חצי עם Trinity עדיין הציגה חלונות של שלוש שנים בהפסד; יש בו גם הנחת ריבית מזומן ונתוני מאקרו שאינם ארכיון פרסומים היסטורי.

**Ladder 4 נשאר אמת מידה חזקה.** באותה בדיקת 2012 ובאותו איזון שנתי, הוא הניב 21.6% עם ירידה של 13.4%, מול 21.6% ו־14.7% באגרסיבי החדש. לכן אין כאן טענה שהחדש טוב יותר בכל מובן. היתרון שלו הוא שלושה PODים ומחזור מסחר נמוך יותר: כ־24.2 כפולות הון בשנה מול 29.8. בעלויות המחמירות התשואה הייתה 18.8% מול 18.1%. ההשוואה משווה הרכבים תחת כלל איזון אחיד; YAML הישן של Ladder 4 עצמו אינו מאוזן שנתית.

![תשואות שנתיות](charts/annual.png)

## סיכון, עלויות ותפעול

מבחני התכנון ההיסטוריים היו: עד 15% ירידה למאקרו, 25% לצמיחה החודשית והפעילה, ו־40% לאגרסיבי; לצד הגבלת שנה גרועה, תקופת התאוששות וחלונות החזקה של 3/5/7 שנים. **אלו תנאי פסילה למחקר, לא מגבלות שמבטיחות לעצור הפסד עתידי.**

בזעזוע היפותטי סימולטני — מניות ‎−20%, אג״ח ארוכות ‎−15%, זהב ‎−10% וסחורות ‎−20% — החודשי הגיע לכ־20.2% הפסד, הפעיל לכ־{abs(active_exposure_dict['worst_liquidation_state'])*100:.1f}% והאגרסיבי לכ־35.2%. במאקרו התקבל חסם שמרני של כ־15.9%; חצי התיק שב־Trinity הוצב בנכס הגרוע בתרחיש משום שאין יומן אחזקות היסטורי מאומת עבורו. אלה זעזועים על מצבי אחזקה, ללא תגובה של האסטרטגיות, ולא הסתברויות או תחזית.

{cost_table_str}

התרחיש המרכזי שומר על עמלות והחלקת המחיר המקוריות, משלים ניכוי דיבידנדים ארוכים ל־25% ומחייב מימון חסר ב־5% לשנה. המחמיר מוסיף 10 נקודות בסיס לכל צד מסחר, מעלה את המימון ל־8% ואת עלות השורט עד ל־5%. אלו התאמות על אחזקות שנשמרו, לא הרצה מחדש של כל גודל פוזיציה. נכללת עלות נוספת של 10 נקודות בסיס בהקצאה הראשונית ועל היקף שינוי משקולות שנתי. תרחיש דמי ניהול של 1% נבדק בנפרד; העלויות אינן תמחור ברוקר ללקוח מסוים.

״חודשי״ פירושו החלטות ומסחר **מתוכננים** בחודש. במקור MOSAIC היו גם 71 יציאות כפויות בגלל מחיר חסר או סיום מסחר בנייר בחלון הארוך. זהו מודל סילוק של המנוע, לא הוכחה לטיפול אמיתי ומדויק באירועי חברה. פעולות כאלה מחייבות מעקב גם בין מועדי האיזון. בשאר שלושת המוצרים יש החלטות או מסחר יומיים.

![תלות מתגלגלת בשוק](charts/rolling_correlation.png)

## רמת הביטחון והחלטה מעשית

השתמשתי בכל 25 האסטרטגיות כמועמדות, ובנתוני החשבון המקוריים שלהן. לא נוצרה תשואה לפני תחילת היסטוריה של POD, ולא מולאו תשואות חסרות. נבדקו 73 תצורות מוצהרות בשלב המקורי, כולל בקרות ושכנויות שחלקן חופפות, אחר כך בדיקות שבריריות ומועמד פעיל אחד. כל היסטוריית האסטרטגיות כבר שימשה מחקר בעבר. אין כאן מדגם מבחן בלתי נגוע; תיקון סטטיסטי לשש השוואות אינו מוחק את חיפוש האסטרטגיות שקדם למחקר. זו בדיוק הבעיה המתוארת ב[מחקר על התאמת־יתר בתיקי מניות](https://carmamaths.org/resources/jon/stockfund.pdf).

כבר קראתי את מסמך Claude לפני בקשתך האחרונה. לא ניתן למחוק את החשיפה הזאת. במחקר הנוכחי לא קראתי שוב את מוצריו, משקולותיו או קוד בנייתם. עמית ביקורת עצמאי גיבש מנדטים מתוך מקורות האסטרטגיות בלבד. גם תוצאות המחקר הקודם שלי מוכרות ולכן אינן מוצגות כאימות עצמאי.

**ההמלצה היא להציג תפריט מטרות והרכבים, ולא למכור את אחוזי העבר כיעדים.** שלושת המועמדים הראשיים מספקים נקודת פתיחה ברורה; הפעיל הוא אפשרות משנית למי שמעדיף מיתון ימי הפסד ומקבל את מגבלות הראיות. ללקוח שדורש שמירת קרן, שימוש קרוב בכסף או מסחר אפסי מחוץ למועד קבוע — אין במחקר מוצר מתאים שהוכח.

לפני הפעלה ללקוח מסוים נדרשים התאמה למטבע ולאופק שלו, הרצה של התיק המלא בהון ובעלויות שלו, כימות כמותי של עיגול יחידות ופדיונות, ותיקוף ביצוע ו־capacity. PM_READY ו־WIRED הם מצב תשתיתי, לא אישור התאמה ללקוח. כל אלה נפרדים מהמחקר שהושלם כאן.

[נספח מלא: כל המועמדים, כללים, משקולות, כשלים ומקורות](REPORT_FULL.html) · [טבלת ההשוואה](tables/menu_common_sample.csv) · [פרוטוקול קפוא](research_spec_frozen.json)
'''
    narrative_str = narrative_str.replace("ובנתוני החשבון המקוריים שלהן", "וביומני החשבון בסימולציות המקוריות שלהן")
    (STUDY_PATH/"REPORT.md").write_text(narrative_str, encoding="utf-8")
    (STUDY_PATH/"REPORT.html").write_text(render_html(narrative_str, "תפריט תיקים עצמאי — Alpha Super"), encoding="utf-8")
    catalog_list = json.loads((SOURCE_PATH/"catalog_complete.json").read_text(encoding="utf-8"))
    rule_block_list = []
    for row_dict in catalog_list:
        alias_str = row_dict["alias"]
        source_id_str = row_dict["strategy_import"].split(":")[0].split(".")[-1]
        metadata_dict = json.loads((SOURCE_PATH/"data"/source_id_str/"source_metadata.json").read_text())
        rule_block_list.append(f'<details><summary>{html.escape(alias_str)} · {html.escape(row_dict["friendly_name_he"])} · {row_dict["tier"]}</summary><p class="source">{html.escape(row_dict["strategy_import"])}</p><p>{metadata_dict["actual_start_date_str"]} — {metadata_dict["actual_end_date_str"]}; native capital: ${metadata_dict["native_capital_float"]:,.0f}</p><pre>{html.escape(json.dumps({key_str:row_dict.get(key_str) for key_str in ["family", "mechanism", "cadence", "universe", "rules", "target_exposure", "execution_timing", "costs", "cash_policy", "caveats", "evidence"]}, ensure_ascii=False, indent=2))}</pre></details>')
    weight_rows_list = []
    alias_by_name_dict = {row_dict["alias"]: row_dict["strategy_import"] for row_dict in catalog_list}
    for candidate_str in ["S2", "M0", "A3", "G2"]:
        for alias_str, weight_float in weight_dict[candidate_str].items():
            weight_rows_list.append(f"| {HEBREW_DICT[candidate_str]} | {alias_str} | {weight_float:.12f} | `{alias_by_name_dict[alias_str]}` |")
    complete_str = '''# נספח מחקר ומסלול ביקורת

**סיכום:** זהו נספח ל[דוח ההחלטה](REPORT.html). כל ההרכבים הם הצעות מחקר. אף הגדרת הקצאה במערכת הייצור לא שונתה.

## משקולות ומזהים מדויקים

| תיק | שם קצר | משקל הון | מזהה מקור |
|---|---|---:|---|
''' + "\n".join(weight_rows_list) + '''

## חוזה החישוב

```text
Client purpose -> eligible mechanisms -> fixed capital weights
                         |
                         v
Selected menu: Close_T decision -> Open_(T+1) execution
*** CRITICAL *** no data after decision boundary enter orders
                         |
                         v
Independent sleeve NAVs -> annual synthetic transfers -> portfolio NAV
                         |
                         v
Risk / costs / role ablation / adverse regimes -> menu decision
```

מודל יחידות POD:

```text
r_i,t = NAV_i,t / NAV_i,t-1 - 1
E_i,t = E_i,t-1 * (1 + r_i,t)                between annual resets
E_portfolio,t = sum_i E_i,t
w_i,t-1 = E_i,t-1 / E_portfolio,t-1
cost_t = 0.001 * sum_i abs(target_i - w_i,t-1) on annual reset
E_i,t(before return) = E_portfolio,t-1 * (1-cost_t) * target_i
initial cost = 0.001
DD_t = E_t / max(E_0,...,E_t) - 1
CAGR = (E_N / E_0)^(252/N) - 1
beta = Cov(r_portfolio,r_SPY) / Var(r_SPY)
ES5_loss = -mean(r_t | r_t <= empirical_5th_percentile)
```

העוגן הוא סגירה נצפית של חשבונות קיימים, לא פיקדון חדש שמייצר בבת אחת את כל האחזקות. האיזון משתמש במשקולות סוף היום הקודם לפני קריאת תשואת היום. זהו מודל העברת יחידות מחקר, לא יומן פקודות פיזי בין PODים. אין שינוי בחוקי האסטרטגיות.

תשואות, דיבידנדים, עמלות ושורט נלקחו מיומני המקור. חוסר תשואה או אי־התאמה בין תאריכי המקור גורמים לכשל. בתצלומי אחזקות בלבד, תא ריק של נכס שאינו מוחזק הוחלף באפס. שיעורי Sharpe מוצגים עם ריבית אפס ובמקביל כעודף מול BIL; אין להשוות Sharpe מזומן בתנודתיות זעירה ליתרון עסקי. CAGR משתמש ב־252 תשואות לשנה, ומבחני האופק משתמשים ב־756/1260/1764 ימי מסחר. תקופה שטרם התאוששה נשארת מצונזרת, ולא מוצגת כהתאוששות.

מימון נוסף נגבה על max(collateral-cash,0); collateral מדויק מיומן שורט כאשר קיים, אחרת קירוב 102% לפי שווי שורט. דיבידנדים ארוכים הושלמו ל־25% ניכוי; אין חיוב כפול במס או בריבית שכבר נמצאים ב־NAV. רק צבירה שמיוחסת לתקופה שמעבר לסגירת ההשוואה הוחזרה. Crisis Trend גובה לפי יום מסחר נוכחי ולכן אין לו החזר כזה. ריבית המזומן של Tactical Fixed Income היא חלק מהמקור; הסרתה היא רגישות נפרדת.

רוב המקורות הם חשבונות היסטוריים של $100,000, ושני מדדי ETF חושבו בחשבונות $1,000,000. הסקלה המדווחת אינה מוכיחה עיגול מניות, עמלות מינימום או capacity עבור הון אחר. מודל Sector6 הוא עם שברי מניות. בבחינת חשיפה עורבבו אחזקות מקור עם משקולות יחידות שהותאמו לעלויות; זהו תיאור אינדיקטיבי של סיכון, לא חשבון מסחר ששוחזר במלואו.

## מקור הנתונים ומגבלות

25 המקורות נשמרו קודם למחקר החדש ונבדקו מול manifest והאש של כל קובץ. מניות משתמשות ביקום Norgate היסטורי; מחירי ביצוע וסימון CAPITALSPECIAL עם דיבידנדים מפורשים. אותות TOTALRETURN קיימים היכן שחוקי המקור דורשים זאת. FRED בחלק מהמודלים הוא מהגרסה הזמינה כיום, לא ALFRED. אין היסטוריה מלאה זהה לכל POD: החלון לכל25 מתחיל לאחר4.4.2019; חלון ההשוואה המשותף לארבעת מוצרי התפריט מתחיל לאחר19.7.2018; החלון הארוך לשלושת המוצרים הראשיים מתחיל לאחר1.10.2012 ומסתיים31.7.2026. אין טענה לעדכון ביצועים עד יום הדוח בספטמבר.

הזהות של קוד המקור נבדקה כיום, אך אין בכל הריצות הישנות hash שמוכיח שזה היה קוד הריצה בזמן ההפקה. Trinity חסר יומן אחזקות אמיתי שמור. ניסיון לשחזר עסקאות מול מחירי Norgate עדכניים נכשל בסף התאמה: פער מרבי0.294%מה־NAV. הנתונים המשוחזרים לא שולבו. מדד gross הכולל שלו נלקח מהחשבונאות המקורית, ובתרחישי נכסים השתמשנו במפורש בחסם הפסד שמרני.

## מנדטים והחלטות

| שימוש | סף ירידה היסטורית | שנה גרועה | זמן מתחת לשיא | אין הפסד בחלון |
|---|---:|---:|---:|---:|
| מאקרו מתון | 15% | 10% | 756 ימי מסחר | 3 שנים |
| חודשי | 25% | 20% | 1260 ימי מסחר | 5 שנים |
| פעיל | 25% | 20% | 1008 ימי מסחר | 5 שנים |
| אגרסיבי | 40% | 30% | 1764 ימי מסחר | 7 שנים |

הפעיל חייב לשפר ירידה מרבית בשתי נקודות אחוז או להקטין ES5 ב־10%, תוך ויתור של עד1.5נקודות אחוז תשואה מול החודשי. האגרסיבי חייב להוסיף לפחות2נקודות אחוז תשואה לעומת בסיס פעיל, וגם להשתפר בשני שלישים מן הבלוקים הכרונולוגיים; מינוף חייב להצדיק את עצמו גם מול הבסיס האגרסיבי ללא ETF ממונף. קביעת ספים לפני הבדיקה מונעת שינוי נוח אחרי תוצאה, אך אינה הופכת את הספים לאמיתות טבע או לעדות מחוץ למדגם.

A3 הוא הרחבה אחרי צפייה בהיסטוריית2019+. אותו הרכב נבחן עד תחילת ההיסטוריה האמיתית ביולי2018, עם אותן דרישות הפסד ותועלת, הפעם גם תחת עלויות מחמירות. הוא עבר, אך לא זכה למעמד של אימות עצמאי או להיסטוריה מ־2012.

## טבלאות מלאות

- [כל התוצאות המקוריות](tables/results.csv)
- [פסילת מנדטים](tables/mandate_gates.csv)
- [כל25 המקורות על אותו חלון](tables/universe.csv)
- [קורלציה רגילה](tables/correlation.csv) ו[קורלציה בימי SPY הגרועים](tables/tail_correlation.csv)
- [תרומת כל POD לסיכון](tables/risk_contributions.csv)
- [בדיקות משקולות והסרת רכיבים](tables/fragility.csv)
- [המועמד הפעיל והחלופה החודשית](tables/active_alternative.csv)
- [חלונות היסטוריים ומשברים](tables/periods.csv)
- [תדירות מסחר מתוכנן ויציאות כפויות](tables/cadence.csv)
- [חשיפות ומחזור מסחר](tables/exposures.csv)
- [זעזועים היפותטיים](tables/shock_scenarios.csv)
- [השוואות bootstrap](tables/inference.csv)

טבלת אירועים שומרת גם CAGR מחושב אוטומטית; לא משתמשים בו לפרשנות של משבר קצר. יש לקרוא total_return באירוע. Ladder1 ו־Ladder4 נבנו מן המשקולות הישנות תחת איזון שנתי אחיד לצורך ההשוואה; זו אינה העתקה מלאה של YAML שלא מגדיר איזון. כללי bootstrap:2000דגימות של בלוקים מעגליים בני63ימים, זרע20260923; רווח95%לממוצע תשואה שנתי מזווג; Holm על שש השוואות מוצהרות. זו אינה הסתברות לתשואה עתידית, ואינה מתקנת את כל חיפוש הפורטפוליו או את היסטוריית חיפוש האסטרטגיות.

## מיפוי כל25 האסטרטגיות

כל25האסטרטגיות הופיעו לפחות במועמד או בבדיקת החלפה/הגנה. שבע גרסאות Defense First אינן שבעה מנגנונים נפרדים. משפחות HPI/QPI/DV2 ומומנטום NDX/MOSAIC חולקות סיכוני מניות. Compass הוא מחזור סקטורים, לא ביטוח; MonthEndFlow הוא מסחר בלחץ איזון, לרבות שורט ו־MOC, ולא מוצר חודשי פשוט. VIXM מגיב למצב תנודתיות ואינו מבטיח הגנה על הפער הראשון. פירוט כללים מלא להלן משמר את חוזה המקור.

''' + "\n".join(rule_block_list) + '''

## תוצרי שחזור

- [פרוטוקול ראשוני](research_spec_frozen.json)
- [הוספת היסטוריית2008 לפני התוצאות](amendment_001_before_results.json)
- [הרחבת בדיקות משקולות אחרי התוצאות](amendment_002_fragility.json)
- [היפותזת המועמד הפעיל](adaptive_active_spec.json)
- [מחברת החלטה שבוצעה](decision_notebook.ipynb)
- [מצב המחקר](research_state.json), [יומן ניסויים](experiment_ledger.jsonl), [יומן החלטות](decision_log.jsonl)
- [אימות וביקורת](verification/FINAL_VERIFICATION.md)
- [manifest](run_manifest.json)

המחקר מוגבל לתכנון ומדידה. לא שונו הקצאות, אסטרטגיות, מנוע, scheduler, broker או הגדרות LIVE. אין צורך לפרסם ל־Notion או למערכת חיצונית כדי לשחזר את התוצאה.
'''
    (STUDY_PATH/"REPORT_FULL.md").write_text(complete_str, encoding="utf-8")
    (STUDY_PATH/"REPORT_FULL.html").write_text(render_html(complete_str, "נספח תפריט תיקים עצמאי"), encoding="utf-8")
    print(json.dumps({"main_report": str(STUDY_PATH/"REPORT.html"), "main_words": len(narrative_str.split()), "matched_menu": central_df.loc[["S2", "M0", "A3", "G2"], ["cagr", "max_drawdown"]].to_dict("index"), "active_exposure": active_exposure_dict}, ensure_ascii=True))


if __name__ == "__main__":
    main()
