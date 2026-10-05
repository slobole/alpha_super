"""Build the defensive v2 report page: template + report_data.json (+ optional review HTML) -> report/defensive_v2.html."""

from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[2] / "results" / "research" / "portfolio" / "defensive_v2_20260929" / "report"

html = (HERE / "report_template.html").read_text(encoding="utf-8")
data = (REPORT / "report_data.json").read_text(encoding="utf-8")
review_path = REPORT / "review.html"
review = json.dumps(review_path.read_text(encoding="utf-8")) if review_path.exists() else "null"
html = html.replace("/*DATA*/null", data.replace("</", r"<\/")).replace("/*REVIEW*/null", review.replace("</", r"<\/"))
(REPORT / "defensive_v2.html").write_text(html, encoding="utf-8")
print(len(html))
