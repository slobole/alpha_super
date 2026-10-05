"""Reviewer scratch (report lens): language-rule scan of the built page (read-only).
1. RTL blocks whose first strong character is Latin with no RLM before it.
2. Bare digits in Hebrew text outside an LTR span.
3. Hebrew characters inside LTR-only containers."""
import io, re, sys, unicodedata
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/report_lens"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else WT / "results/research/portfolio/fund_products_20261005/report/fund_products_v5.html"
soup = BeautifulSoup(src.read_text(encoding="utf-8"), "lxml")
main = soup.find("main")
RLM = "\u200f"
HEB = re.compile(r"[\u0590-\u05FF]")
LAT = re.compile(r"[A-Za-z]")

def is_ltr(el: Tag) -> bool:
    cls = el.get("class") or []
    return "ltr" in cls or "code" in cls or "mix" in cls or "kpi" in cls or "legend" in cls

def rtl_block(el: Tag) -> bool:
    """True when the element is a block that renders in RTL context."""
    cls = el.get("class") or []
    if is_ltr(el):
        return False
    if el.name in ("td", "th"):
        return "nm" in cls or "he" in cls
    # inside a table but not an RTL cell -> LTR (.tbl table { direction: ltr })
    for p in el.parents:
        if isinstance(p, Tag):
            pc = p.get("class") or []
            if p.name in ("td", "th"):
                return ("nm" in pc or "he" in pc) and not is_ltr(el)
            if is_ltr(p):
                return False
    return True

BLOCKS = ("p", "li", "h1", "h2", "h3", "summary", "td", "th", "div")
viol_start, viol_digits = [], []
for el in main.find_all(BLOCKS):
    cls = " ".join(el.get("class") or [])
    if el.name == "div" and not any(c in cls for c in ("why", "kicker", "cap", "note", "eyebrow")):
        continue
    if not rtl_block(el):
        continue
    # own text: skip nested block children (div.cap inside td.nm, tables inside details, etc.)
    parts = []
    def walk(node, ltr):
        for ch in node.children:
            if isinstance(ch, NavigableString):
                parts.append((str(ch), ltr))
            elif isinstance(ch, Tag):
                if ch.name in ("div", "table", "p", "ul", "ol", "details", "h3"):
                    continue
                walk(ch, ltr or is_ltr(ch))
    walk(el, False)
    text = "".join(t for t, _ in parts)
    stripped = text.lstrip(" \n\t")
    if not stripped:
        continue
    if not HEB.search(text):
        # an RTL block with no Hebrew at all: flag only if it has Latin (renders, but check intent)
        continue
    # 1. first strong char
    first = stripped[0]
    if first != RLM:
        for ch in stripped:
            if HEB.match(ch):
                break
            if LAT.match(ch):
                viol_start.append((el.name, cls, re.sub(r"\s+", " ", stripped)[:110]))
                break
    # 2. bare digits outside ltr spans
    bare = "".join(t for t, l in parts if not l)
    nums = re.findall(r"[+\-−]?\$?\d[\d,.:/]*%?", bare)
    if nums:
        viol_digits.append((el.name, cls, nums, re.sub(r"\s+", " ", stripped)[:90]))

with io.open(OUT / "bidi_scan.txt", "w", encoding="utf-8") as f:
    f.write(f"source: {src}\n\n== RTL blocks that start with Latin and no RLM: {len(viol_start)}\n")
    for v in viol_start:
        f.write(f"<{v[0]} class='{v[1]}'> {v[2]}\n")
    f.write(f"\n== RTL blocks with bare digits outside an LTR span: {len(viol_digits)}\n")
    for v in viol_digits:
        f.write(f"<{v[0]} class='{v[1]}'> {v[2]} :: {v[3]}\n")
    # 3. Hebrew inside LTR containers
    f.write("\n== Hebrew inside LTR containers\n")
    for el in main.find_all(True):
        if is_ltr(el) and HEB.search(el.get_text()):
            f.write(f"<{el.name} class='{' '.join(el.get('class') or [])}'> {re.sub(r'\s+', ' ', el.get_text())[:120]}\n")
    for el in main.find_all(["td", "th"]):
        cls = el.get("class") or []
        if "nm" in cls or "he" in cls:
            continue
        own = "".join(str(c) for c in el.children if isinstance(c, NavigableString))
        if HEB.search(el.get_text()):
            f.write(f"<{el.name} class='{' '.join(cls)}'> (LTR table cell with Hebrew) {re.sub(r'\s+', ' ', el.get_text())[:120]}\n")
print("start violations", len(viol_start), "digit blocks", len(viol_digits))
