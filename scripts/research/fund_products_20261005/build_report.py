"""Fund products report v5 (Hebrew page, English statistics): DEFENSIVE carried from the 2026-10-01 study, GROWTH
rebuilt from three capsules (fund products final pass, 2026-10-05).

Reads <study>/report/{study,battery,capacity,exposure,defensive,versus,monthly,pm_confirm}.json and the stored defensive rows
(fund_products_20260930/report/report_a6.json); writes <study>/report/fund_products_v5.html. Tables are rendered here;
the page's script only draws the charts. Texts live in report_texts.py.

Usage: python build_report.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import report_texts as T

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
REP = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "report"
OLD = WT / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
TEMPLATE = HERE.parent / "fund_products_20260930" / "report_template.html"

POD = {"taa3x": "TAA 3x", "taa3x_1n": "TAA 3x 1N", "ndx_vxn": "NDX-VXN", "core5": "CORE5", "btal_qqq": "BTAL_QQQ", "tbill": "BIL",
       "dv2": "DV2", "hpi_vote": "HPI", "etf_dv2": "DV2-IND", "eom_flow": "EOM", "downshock": "downshock", "ndx_atr_cap": "NDX ATR cap",
       "ndx_natr_cap": "NDX NATR cap", "dv2_g": "DV2-G", "hpi_g": "HPI-G", "qqq_tr": "QQQ", "debt": "margin loan"}
CAPS = {"taa3x": "TAA 3x", "taa3x_1n": "TAA 3x 1N", "ndx_atr_cap": "MOM", "ndx_natr_cap": "MOM", "dv2_g": "MR", "hpi_g": "MR"}


def load(name: str, folder: Path = REP) -> dict:
    p = folder / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x, d: int = 1) -> str:
    return "–" if x is None else f"{x * 100:.{d}f}%"


def sg(x, d: int = 1) -> str:
    if x is None:
        return "–"
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x) * 100:.{d}f}%"


def f2(x, d: int = 2) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def usd(x, top: bool = False) -> str:
    if not x:
        return "–"
    return ("≥" if top else "") + "$" + (f"{round(x / 1e6, 1):g}M" if x >= 1e6 else f"{x / 1e3:.0f}K")


def mix(w: dict) -> str:
    """Capital mix at capsule level (the MOM and MR pods are shown as one capsule each)."""
    agg: dict = {}
    for k, v in w.items():
        name = CAPS.get(k, POD.get(k, k))
        agg[name] = agg.get(name, 0.0) + v
    order = ["TAA 3x", "TAA 3x 1N", "MOM", "NDX-VXN", "MR", "DV2", "HPI", "CORE5", "BTAL_QQQ", "DV2-IND", "EOM", "downshock", "QQQ", "BIL"]
    rank = lambda k: order.index(k) if k in order else len(order)  # noqa: E731
    return " · ".join(f"{v * 100:.0f}% {k}" for k, v in sorted(agg.items(), key=lambda kv: (-round(kv[1], 6), rank(kv[0]))))


PIE_ORDER = ["TAA 3x", "TAA 3x 1N", "MOM", "NDX-VXN", "MR", "DV2", "HPI", "CORE5", "BTAL_QQQ", "DV2-IND", "EOM", "downshock", "QQQ", "BIL"]
PIE_NAME = {"MOM": "Momentum capsule", "MR": "MR capsule", "BIL": "Cash (BIL)"}
PIE_COLOR = {"TAA 3x": "var(--s3)", "TAA 3x 1N": "var(--s3)", "MOM": "var(--s1)", "NDX-VXN": "var(--s1)", "MR": "var(--s4)", "DV2": "var(--s4)", "HPI": "var(--s4)",
             "CORE5": "var(--s2)", "BTAL_QQQ": "var(--s5)", "DV2-IND": "var(--warn)", "EOM": "var(--accent)", "downshock": "var(--bad)", "QQQ": "var(--bench)", "BIL": "var(--ink-3)"}


def pie(w: dict) -> str:
    """Capital mix as a donut with its legend (capsule level, same grouping as mix())."""
    import math  # noqa: PLC0415
    agg: dict = {}
    for k, v in w.items():
        name = CAPS.get(k, POD.get(k, k))
        agg[name] = agg.get(name, 0.0) + v
    tot = sum(agg.values())
    rank = lambda k: PIE_ORDER.index(k) if k in PIE_ORDER else len(PIE_ORDER)  # noqa: E731
    items = sorted(((k, v / tot) for k, v in agg.items() if v / tot > 0.004), key=lambda kv: (-round(kv[1], 6), rank(kv[0])))
    cx = cy = 50.0
    ro, ri = 46.0, 27.0
    pt = lambda r, a: f"{cx + r * math.sin(a):.2f} {cy - r * math.cos(a):.2f}"  # noqa: E731
    paths, ang = [], 0.0
    for k, v in items:
        col = PIE_COLOR.get(k, "var(--ink-2)")
        if v > 0.999:
            paths.append(f'<circle cx="50" cy="50" r="{(ro + ri) / 2}" fill="none" stroke="{col}" stroke-width="{ro - ri}"/>')
            break
        a0, a1 = ang, ang + v * 2 * math.pi
        big = 1 if v > 0.5 else 0
        paths.append(f'<path d="M {pt(ro, a0)} A {ro} {ro} 0 {big} 1 {pt(ro, a1)} L {pt(ri, a1)} A {ri} {ri} 0 {big} 0 {pt(ri, a0)} Z" fill="{col}" stroke="var(--surface)" stroke-width="1.5"/>')
        ang = a1
    pc = lambda v: f"{v * 100:.1f}".rstrip("0").rstrip(".") if abs(sum(round(x * 100) for _, x in items) - 100) > 0.5 else f"{v * 100:.0f}"  # noqa: E731
    label = ", ".join(f"{PIE_NAME.get(k, k)} {pc(v)}%" for k, v in items)
    legend = "".join(f'<li><i style="background:{PIE_COLOR.get(k, "var(--ink-2)")}"></i><b>{pc(v)}%</b> {html.escape(PIE_NAME.get(k, k))}</li>' for k, v in items)
    return (f'<div class="pie"><svg viewBox="0 0 100 100" role="img" aria-label="{html.escape(label)}">{"".join(paths)}</svg><ul>{legend}</ul></div>')


def td(v: str, cls: str = "") -> str:
    v = re.sub(r"(?<![\w>\"=])-(?=\d)", "−", v)         # a real minus sign before numbers (not inside words, tags or dates)
    return f'<td class="n {cls}">{v}</td>'


def table(head: list[str], rows: list[str], tid: str = "") -> str:
    hid = f' id="{tid}"' if tid else ""
    return (f'<div class="tbl"><table{hid}><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def tr(cells: list[str], cls: str = "") -> str:
    return f'<tr class="{cls}">' + "".join(cells) + "</tr>"


def name_cell(he: str, sub: str = "") -> str:
    if re.match(r"[A-Za-z0-9]", he):          # a right-to-left cell must not open with a Latin token: RLM first
        he = "\u200f" + he
    return f'<td class="nm">{he}' + (f'<div class="cap">{sub}</div>' if sub else "") + "</td>"


def lt(s: str) -> str:
    return f'<td>{html.escape(s)}</td>'


def yes(b) -> str:
    return '<span class="yes">yes</span>' if b else '<span class="no">no</span>'


def heat(matrix: dict, names: list[str], title: str) -> str:
    """Correlation heat table; colours come from the theme tokens through color-mix."""
    h = f'<div class="tbl"><table><thead><tr><th>{title}</th>' + "".join(f"<th>{n}</th>" for n in names) + "</tr></thead><tbody>"
    for a in names:
        h += f'<tr><td style="white-space:nowrap">{a}</td>'
        for b in names:
            if a == b:
                h += '<td class="c" style="background:var(--surface-2)"></td>'
                continue
            v = matrix[a][b]
            pole = "var(--pos-pole)" if v >= 0 else "var(--neg-pole)"
            h += f'<td class="c" style="background:color-mix(in srgb, {pole} {min(abs(v), 1) * 70:.0f}%, var(--mid))">{v:.2f}</td>'
        h += "</tr>"
    return h + "</tbody></table></div>"


def main() -> int:
    S, B, C, X, D, P = (load(f"{n}.json") for n in ("study", "battery", "capacity", "exposure", "defensive", "pm_confirm"))
    A6 = load("report_a6.json", OLD)
    V = load("versus.json")
    M = load("monthly.json")          # exploratory monthly-book grid (monthly.py); empty until it has run
    stale = lambda name: (REP / name).exists() and (REP / name).stat().st_mtime < (REP / "study.json").stat().st_mtime  # noqa: E731
    if stale("pm_confirm.json"):      # an engine confirmation older than the study belongs to the previous product set: show it as pending
        P = {}
    if stale("monthly.json"):
        M = {}
    FEES = load("fees.json")
    ctx = T.build(S, B, C, X, D, P, A6, V=V, M=M, h=dict(FEES=FEES, pct=pct, sg=sg, f2=f2, usd=usd, mix=mix, td=td, table=table, tr=tr, name_cell=name_cell,
                                             lt=lt, yes=yes, heat=heat, POD=POD, pie=pie))
    head = "\n".join(TEMPLATE.read_text(encoding="utf-8").splitlines()[:104])
    body = (HERE / "report_body.html").read_text(encoding="utf-8")
    page = head + "\n" + body
    for key, val in ctx["html"].items():
        token = f"<!--{key}-->"
        assert token in page, key
        page = page.replace(token, val)
    page = page.replace("/*CHARTS*/null", json.dumps(ctx["charts"], ensure_ascii=False).replace("</", "<\\/"))
    assert "<!--" not in page.replace("<!--[", ""), [ln for ln in page.splitlines() if "<!--" in ln][:3]
    out = REP / "fund_products_v5.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, len(page))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
