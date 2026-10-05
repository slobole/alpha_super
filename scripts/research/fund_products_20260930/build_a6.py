"""Build report/fund_products_v3.html: report_template.html's head + report_a6_body.html + report_a6.json + a6_texts.py.

Usage: python build_a6.py
"""

from __future__ import annotations

import json
from pathlib import Path

import a6_texts as T

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[2] / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"


def main() -> int:
    head = "\n".join((HERE / "report_template.html").read_text(encoding="utf-8").split("\n")[:104]) + "\n"
    body = (HERE / "report_a6_body.html").read_text(encoding="utf-8")
    html = head + body
    data = (REPORT / "report_a6.json").read_text(encoding="utf-8")
    rv = REPORT / "review_a6.html"
    review = json.dumps(rv.read_text(encoding="utf-8")) if rv.exists() else "null"
    for key, val in T.TEXT.items():
        html = html.replace(f"/*{key}*/", val)
    html = html.replace("/*META*/null", json.dumps(T.META, ensure_ascii=False)).replace("/*LABELS*/null", json.dumps(T.LABELS, ensure_ascii=False))
    html = html.replace("/*DATA*/null", data.replace("</", "<\\/")).replace("/*REVIEW*/null", review.replace("</", "<\\/"))
    left = [k for k in ("LEDE", "SUMMARY", "DEF_INTRO", "DS_TITLE", "DS_TEXT", "GRO_INTRO", "CALLOUT_22", "ALPHA_TEXT", "DROPPED", "TODO", "CAVEATS", "META", "LABELS")
            if f"/*{k}*/" in html]
    assert not left, f"unfilled placeholders: {left}"
    (REPORT / "fund_products_v3.html").write_text(html, encoding="utf-8")
    print(len(html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
