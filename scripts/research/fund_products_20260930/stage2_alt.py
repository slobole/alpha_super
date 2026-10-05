"""Amendment A2 (post-result, labelled): stage 2 with alternative TAA blocks (TAA-3X split, TAA-1N single).

Usage: python stage2_alt.py
"""

from __future__ import annotations

import pandas as pd

import fp_lib as fp
from fp_lib import Book, ga, lib
import stage2

ALT = {"TAA3X": {"taa3x": 0.5, "taa3x_1n": 0.5}, "TAA1N": {"taa3x_1n": 1.0}}
OPTS = [o for o in stage2.OPTIONS if o[1] in ("LOW-TOUCH", "MAIN") and o[0] in ("GROWTH", "GROWTH PLUS", "DEFENSIVE")]


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    base = pd.read_csv(fp.STUDY / "main" / "block_returns.csv.gz", index_col=0, parse_dates=True)
    for tag, w in ALT.items():
        blocks = base.copy()
        blocks[f"TAA|{tag}"] = lib.book_returns(frame, Book(tag, tuple(w), "EQ", w), start).reindex(blocks.index)
        cols = {line: {k: (f"TAA|{tag}" if k == "TAA" else v) for k, v in c.items()} for line, c in stage2.BLOCK_COL.items()
                if line in ("LOW-TOUCH", "MAIN")}
        stage2.main("main", block_col=cols, options=OPTS, tag=f"_{tag}", blocks=blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
