"""Read-only: rebuild fund_products_v4.html in memory (same steps as build_v4.py) and compare with the file on disk.
Also list report files with sizes, extract section headings, and measure the v4 page's embedded payload sizes."""
import hashlib
import json
import re
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
SRC = WT / "scripts" / "research" / "fund_products_20260930"
REP = WT / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "report_map"
sys.path.insert(0, str(SRC))
import v4_texts as T  # noqa: E402

lines = []
head = "\n".join((SRC / "report_template.html").read_text(encoding="utf-8").split("\n")[:104]) + "\n"
html = head + (SRC / "report_v4_body.html").read_text(encoding="utf-8")
data = (REP / "report_a6.json").read_text(encoding="utf-8")
a7 = json.loads((REP / "a7.json").read_text(encoding="utf-8"))
a7_small = {"targets": a7["targets"], "unlevered": a7["unlevered"], "levered": a7["levered"],
            "rows": {k: {f: v[f] for f in ("L", "cagr", "plus10", "dd", "reg_t", "tails", "t")} for k, v in a7["rows"].items()}}
review = json.dumps((REP / "review_v4.html").read_text(encoding="utf-8"))
for key, val in T.TEXT.items():
    html = html.replace(f"/*{key}*/", val)
html = html.replace("/*META*/null", json.dumps(T.META, ensure_ascii=False)).replace("/*LABELS*/null", json.dumps(T.LABELS, ensure_ascii=False))
html = html.replace("/*DATA*/null", data.replace("</", "<\/")).replace("/*A7*/null", json.dumps(a7_small).replace("</", "<\/"))
html = html.replace("/*REVIEW*/null", review.replace("</", "<\/"))
disk = (REP / "fund_products_v4.html").read_text(encoding="utf-8")
lines.append(f"rebuilt chars {len(html)} ; disk chars {len(disk)} ; identical {html == disk}")
lines.append(f"disk bytes {(REP / 'fund_products_v4.html').stat().st_size} sha256 {hashlib.sha256((REP / 'fund_products_v4.html').read_bytes()).hexdigest()[:16]}")
lines.append(f"payload chars: DATA {len(data)} ; A7 small {len(json.dumps(a7_small))} ; head {len(head)} ; body {len((SRC / 'report_v4_body.html').read_text(encoding='utf-8'))}")
lines.append("external refs in v4: " + str(sorted(set(re.findall(r'(?:href|src)="(https?://[^"]+)"', disk)))))
lines.append("script tags: " + str(re.findall(r"<script[^>]*>", disk)))
lines.append("title: " + str(re.findall(r"<title>(.*?)</title>", disk)))
lines.append("has <html>/<head>/<body> tags: " + str([t for t in ("<html", "<head", "<body", "<!doctype", "<!DOCTYPE") if t in disk]))
lines.append("placeholders in body: " + str(re.findall(r"/\*([A-Z0-9_]+)\*/", (SRC / "report_v4_body.html").read_text(encoding="utf-8"))))
lines.append("TEXT keys: " + str(list(T.TEXT)))
lines.append("RLM count in v4_texts TEXT values: " + str(sum(v.count("\u200f") for v in T.TEXT.values())) + " ; LRM count: " + str(sum(v.count("\u200e") for v in T.TEXT.values())))
# Hebrew line-start check: any TEXT <li>/<p> fragment whose first visible char is Latin (no RLM before)
bad = []
for k, v in T.TEXT.items():
    for frag in re.split(r"<li>|<p>", v):
        vis = re.sub(r"<[^>]+>", "", frag).lstrip()
        if vis and re.match(r"[A-Za-z0-9$]", vis[0]):
            bad.append((k, vis[:40]))
for k, r in T.ROWS.items():
    for f in ("name", "why"):
        vis = re.sub(r"<[^>]+>", "", r.get(f, "")).lstrip()
        if vis and re.match(r"[A-Za-z0-9$]", vis[0]):
            bad.append((k + "." + f, vis[:40]))
lines.append("texts starting with a Latin char without RLM: " + str(bad))
lines.append("== report dir files")
for p in sorted(REP.iterdir()):
    lines.append(f"{p.name:28s} {p.stat().st_size:>9d}")
(OUT / "rebuild_check.txt").write_text("\n".join(lines), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(lines))
