import re, html, sys
from pathlib import Path
R = Path("results/research/portfolio/fund_products_20261005")
for tag in ("report", "report_equal_capital_snapshot"):
    t = (R/tag/"fund_products_v5.html").read_text(encoding="utf-8")
    t = re.sub(r"<script.*?</script>", " [SCRIPT] ", t, flags=re.S); t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<svg.*?</svg>", " [SVG] ", t, flags=re.S)
    t = re.sub(r"</(tr|p|div|h\d|li|table|section|summary|details|figcaption)>", "\n", t); t = re.sub(r"</t[dh]>", " | ", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t)); t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
    (Path(__file__).parent/f"page_{tag}.txt").write_text(t, encoding="utf-8")
    print(tag, len(t), t.count("\n"))
