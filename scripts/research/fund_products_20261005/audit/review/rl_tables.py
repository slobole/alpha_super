"""Reviewer scratch: dump every table of fund_products_v5.html as pipe-separated rows (read-only)."""
import io, re
from pathlib import Path
from bs4 import BeautifulSoup

WT = Path(__file__).resolve().parents[5]
REP = WT / "results/research/portfolio/fund_products_20261005/report/fund_products_v5.html"
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/report_lens"
soup = BeautifulSoup(REP.read_text(encoding="utf-8"), "lxml")
clean = lambda s: re.sub(r"\s+", " ", s).strip()
with io.open(OUT / "tables.txt", "w", encoding="utf-8") as f:
    for i, t in enumerate(soup.find_all("table")):
        # nearest preceding heading
        h = t.find_previous(["h2", "h3", "summary"])
        f.write(f"\n===== TABLE {i} (after: {clean(h.get_text()) if h else ''}) =====\n")
        for r in t.find_all("tr"):
            cells = [clean(c.get_text(" ")) for c in r.find_all(["th", "td"])]
            f.write(" | ".join(cells) + "\n")
print("tables", len(soup.find_all("table")))
