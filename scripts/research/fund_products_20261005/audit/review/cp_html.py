"""Compliance reviewer: plain-text view of the report page (tables as rows) for searching what is and is not printed."""
import re, html, sys
from pathlib import Path
p = Path("results/research/portfolio/fund_products_20261005/report/fund_products_v5.html")
t = p.read_text(encoding="utf-8")
body = t[t.index("<main"):t.index("</main>")]
body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=re.S)
body = re.sub(r"</(tr|p|li|h1|h2|h3|div|summary|section)>", "\n", body)
body = re.sub(r"</(td|th)>", " | ", body)
body = re.sub(r"<[^>]+>", "", body)
body = html.unescape(body).replace("\u200f", "")
body = re.sub(r"[ \t]+", " ", body)
body = re.sub(r"\n\s*\n+", "\n", body)
out = Path("results/research/portfolio/fund_products_20261005/audit/review/compliance/report_text.txt")
out.write_text(body, encoding="utf-8")
print(len(body), "chars,", body.count("\n"), "lines")
