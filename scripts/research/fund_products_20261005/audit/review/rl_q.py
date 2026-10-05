"""Reviewer scratch: evaluate a python expression over the report JSONs (read-only). S,B,C,X,D,A6 available."""
import json, sys
from pathlib import Path
WT = Path(__file__).resolve().parents[5]
REP = WT / "results/research/portfolio/fund_products_20261005/report"
OLD = WT / "results/research/portfolio/fund_products_20260930/report"
ld = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
S, B, C, X, D = (ld(REP / f"{n}.json") for n in ("study", "battery", "capacity", "exposure", "defensive"))
A6 = ld(OLD / "report_a6.json")
K = S["books"]
src = sys.stdin.read()
exec(src)
