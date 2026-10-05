"""Fund products, final pass: the products against the 2026-10-01 monthly books, by cost level and by sub-period.

Added after the first results on the independent review's finding that the comparison "GR1 against the monthly book"
was shown at model costs and on the full window only (amendment log, entry R1). Paired stationary bootstrap, mean
block 63, 20,000 paths (seeds 0-9), the same resampled rows for both books and BIL: the share of paths on which the
first book has the higher excess Sharpe / the higher CAGR, at 0 / +5 / +10 bps and in blocks A, B, C and RECENT.

Usage: PYTHONDONTWRITEBYTECODE=1 python versus.py   Writes <study>/report/versus.json.
"""

from __future__ import annotations

import json

import g_lib as g
from g_lib import BLOCK_DICT, Lab, lib

PAIRS = [("GR1", g.S9), ("GR2", g.S9), ("GR3", g.S9), ("GR2", "old growth plus"), ("GR3", "old growth plus"),
         ("S8 GR1 75 / DEF 25", "GR1"), ("S1 no momentum", "GR1"), ("S5 MR tilt", "GR1"), ("S4 core + satellites", "GR1"), ("GR1", g.S0), (g.S9, g.S13_OLD), ("GR2", "GR1")]


def main() -> int:
    lab = Lab()
    for n, w in {**g.PRODUCTS, **g.CHALLENGERS, "old growth plus": g.MONTHLY_PLUS}.items():
        lab.add(n, w)
    series = {"main": lambda n: lab.r(n), "plus5": lambda n: lab.r(n, "s3_plus_5bps"), "plus10": lambda n: lab.plus10(n)}
    out: dict = {}
    for a, b in PAIRS:
        row: dict = {"frames": {}, "blocks": {}}
        for fk, fn in series.items():
            ra, rb = fn(a), fn(b)
            row["frames"][fk] = {"a": g.stats(ra, lab.rf), "b": g.stats(rb, lab.rf), **lab.paired(ra, rb)}
        for blk, (lo, hi) in BLOCK_DICT.items():
            row["blocks"][blk] = {}
            for fk in ("main", "plus5"):
                ra, rb = lib.window(series[fk](a), lo, hi), lib.window(series[fk](b), lo, hi)
                row["blocks"][blk][fk] = {"a": g.stats(ra, lab.rf), "b": g.stats(rb, lab.rf), **lab.paired(ra, rb)}
        out[f"{a} | {b}"] = row
        print(a, "vs", b, {fk: round(v["share_xs"], 3) for fk, v in row["frames"].items()},
              {blk: round(v["main"]["share_xs"], 3) for blk, v in row["blocks"].items()}, flush=True)
    (g.OUT / "versus.json").write_text(json.dumps(g.r6(out), indent=1), encoding="utf-8")
    g.ledger("versus_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
