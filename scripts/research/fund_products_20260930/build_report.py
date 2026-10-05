"""Build report/fund_products.html from report_template.html + report_data.json (+ review.html if present)."""

from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[2] / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
html = (HERE / "report_template.html").read_text(encoding="utf-8")
data = (REPORT / "report_data.json").read_text(encoding="utf-8")
rv = REPORT / "review.html"
review = json.dumps(rv.read_text(encoding="utf-8")) if rv.exists() else "null"
html = html.replace("/*DATA*/null", data.replace("</", "<\/")).replace("/*REVIEW*/null", review.replace("</", "<\/"))
(REPORT / "fund_products.html").write_text(html, encoding="utf-8")
print(len(html))
