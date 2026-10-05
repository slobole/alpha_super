"""Reviewer scratch: dump the visible text of fund_products_v5.html block by block (read-only on the report)."""
import re, sys, json, io
from pathlib import Path
from html.parser import HTMLParser

WT = Path(__file__).resolve().parents[5]
REP = WT / "results/research/portfolio/fund_products_20261005/report/fund_products_v5.html"
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/report_lens"
src = REP.read_text(encoding="utf-8")

BLOCK = {"p", "li", "td", "th", "h1", "h2", "h3", "summary", "div", "section", "header", "ul", "ol", "table", "tr", "thead", "tbody", "details", "main", "span_block"}
VOID = {"br", "meta", "link", "img", "input", "hr"}

class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.blocks = []   # (tag, classes, text)
        self.cur = None
        self.in_script = self.in_style = False
        self.errors = []
    def handle_starttag(self, tag, attrs):
        if tag in ("script",): self.in_script = True
        if tag in ("style",): self.in_style = True
        if tag in VOID: return
        a = dict(attrs)
        self.stack.append((tag, a.get("class", ""), self.getpos()))
        if tag in ("p", "li", "td", "th", "h1", "h2", "h3", "summary") or (tag == "div" and a.get("class", "") in ("why", "cap", "kicker", "kicker ltr", "eyebrow ltr", "mix", "kpi", "legend ltr", "note")):
            self.blocks.append([tag, a.get("class", ""), "", self.getpos()[0], []])
    def handle_endtag(self, tag):
        if tag == "script": self.in_script = False
        if tag == "style": self.in_style = False
        if tag in VOID: return
        if not self.stack:
            self.errors.append(("close with empty stack", tag, self.getpos())); return
        if self.stack[-1][0] != tag:
            self.errors.append(("mismatch", tag, self.getpos(), self.stack[-1]))
            # try to recover
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    return
            return
        self.stack.pop()
    def handle_data(self, data):
        if self.in_script or self.in_style: return
        if not self.blocks: return
        # attach to innermost open block element
        open_tags = [t for t, _, _ in self.stack]
        # find the last block whose tag is still open - approximate: most recent block
        b = self.blocks[-1]
        b[2] += data
        b[4].append((data, [c for _, c, _ in self.stack[-2:]]))

p = P(); p.feed(src)
print("parser errors:", len(p.errors)); [print(e) for e in p.errors[:20]]
print("unclosed at end:", p.stack)
with io.open(OUT / "blocks.txt", "w", encoding="utf-8") as f:
    for tag, cls, text, line, _ in p.blocks:
        t = re.sub(r"\s+", " ", text).strip()
        if t: f.write(f"[{line}] <{tag} class='{cls}'> {t}\n")
print("blocks:", len(p.blocks))
