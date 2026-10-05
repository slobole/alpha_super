"""Build the Hebrew report page: report_template.html + ext_section.html + report/*.json -> report/growth_aggressive.html."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[2] / "results" / "research" / "portfolio" / "growth_aggressive_20260930" / "report"
DIVIDER = """<section class="divider"><header><div class="kicker">המחקר הבסיסי · קרן 2/20, נטו, יחס 50:50</div>
<h2>הדוח הראשון, כפי שפורסם</h2><p class="section-note">מכאן והלאה הדוח המקורי: יעד נטו אחרי 2/20, שתי מדרגות, ויחס TAA:NDX קבוע של 50:50.</p></header></section>
"""

html = (HERE / "report_template.html").read_text(encoding="utf-8")
data = (REPORT / "report_data.json").read_text(encoding="utf-8")
html = html.replace("/*DATA*/null", data.replace("</", r"<\/"))
ext_path = REPORT / "ext_data.json"
if ext_path.exists():
    ext = (HERE / "ext_section.html").read_text(encoding="utf-8")
    ext = ext.replace("/*EXTDATA*/null", ext_path.read_text(encoding="utf-8").replace("</", r"<\/"))
    verdict = (HERE / "verdict_section.html").read_text(encoding="utf-8")
    verdict = verdict.replace("/*ALPHADATA*/null", (REPORT / "alpha.json").read_text(encoding="utf-8").replace("</", r"<\/"))
    seeds = REPORT.parent / "ext" / "verdict_seeds.json"
    verdict = verdict.replace("/*VSEEDDATA*/null", seeds.read_text(encoding="utf-8"))
    first = html.index("<section>")
    html = html[:first] + verdict + "\n" + ext + "\n" + DIVIDER + html[first:]
(REPORT / "growth_aggressive.html").write_text(html, encoding="utf-8")
print(len(html))
