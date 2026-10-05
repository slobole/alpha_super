"""Build report/fund_products_v4.html: report_template.html's head + report_v4_body.html + report_a6.json + a7.json + v4_texts.py.

Usage: python build_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import v4_texts as T

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[2] / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"


def main() -> int:
    head = "\n".join((HERE / "report_template.html").read_text(encoding="utf-8").split("\n")[:104]) + "\n"
    html = head + (HERE / "report_v4_body.html").read_text(encoding="utf-8")
    data = (REPORT / "report_a6.json").read_text(encoding="utf-8")
    a7 = json.loads((REPORT / "a7.json").read_text(encoding="utf-8"))
    a7_small = {"targets": a7["targets"], "unlevered": a7["unlevered"], "levered": a7["levered"],
                "rows": {k: {f: v[f] for f in ("L", "cagr", "plus10", "dd", "reg_t", "tails", "t")} for k, v in a7["rows"].items()}}
    rv = REPORT / "review_v4.html"
    review = json.dumps(rv.read_text(encoding="utf-8")) if rv.exists() else "null"
    for key, val in T.TEXT.items():
        html = html.replace(f"/*{key}*/", val)
    html = html.replace("/*META*/null", json.dumps(T.META, ensure_ascii=False)).replace("/*LABELS*/null", json.dumps(T.LABELS, ensure_ascii=False))
    html = html.replace("/*DATA*/null", data.replace("</", "<\\/")).replace("/*A7*/null", json.dumps(a7_small).replace("</", "<\\/"))
    html = html.replace("/*REVIEW*/null", review.replace("</", "<\\/"))
    left = [k for k in ("LEDE", "SUMMARY", "DEF_INTRO", "DEF_CAP", "GRO_INTRO", "GRO_CAP", "A7_INTRO", "A7_NOTES", "ROB_INTRO", "ALPHA_TEXT",
                        "CRISES_CAP", "CORR_CAP", "DROPPED", "TODO", "CAVEATS", "META", "LABELS", "A7") if f"/*{k}*/" in html]
    assert not left, f"unfilled placeholders: {left}"
    (REPORT / "fund_products_v4.html").write_text(html, encoding="utf-8")
    print(len(html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
