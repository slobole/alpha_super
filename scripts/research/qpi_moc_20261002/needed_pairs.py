"""Which (date, symbol) 15:45 states the QPI test needs, and which are not yet downloaded.

needed[t, i] = possible 15:45 entry: member & final r3 < +1% & Close > 0.95 * SMA200   (generous superset of
               QPI < 30 & r3 < 0 & P > SMA200 at 15:45)
             | held at close t-1 or t in any run listed in held_*.npy (written by run_test.py; exits need data)
Attempted pairs (DV2 study files or ours, with or without data) count as downloaded.
Usage: python needed_pairs.py  -> needed_missing.npz + printed coverage
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402

A0, A1 = "2016-01-04", "2026-09-24"


def attempted(p):
    have = np.zeros(p.C.shape, dtype=bool)
    col = {s: i for i, s in enumerate(p.symbols)}
    for d in (q.REPO / "results/research/dv2_deep_20260925/alpaca/sessions", q.OUT / "alpaca" / "sessions"):
        for f in sorted(d.glob("*.csv.gz")):
            df = pd.read_csv(f, usecols=["norgate_symbol", "date"])
            t = p.dates.get_indexer(pd.to_datetime(df["date"].iloc[:1]))[0] if len(df) else -1
            if t < 0:
                continue
            c = df["norgate_symbol"].map(col).dropna().astype(int).to_numpy()
            have[t, c] = True
    return have


def main():
    p = q.rp.Panel("sp500")
    C = np.asarray(p.C)
    with np.errstate(invalid="ignore", divide="ignore"):
        r3 = C / q.lag(C, 3) - 1.0
        need = np.asarray(p.member) & (r3 < 0.01) & (C > 0.95 * q.rp.sma(p, 200))
    for f in q.OUT.glob("held_*.npy"):
        h = np.load(f)
        need |= h
        need[1:] |= h[:-1]
    m = (p.dates >= A0) & (p.dates <= A1)
    need &= m[:, None]
    have = attempted(p)
    miss = need & ~have
    rows, cols = np.nonzero(miss)
    np.savez(q.OUT / "needed_missing.npz", rows=rows, cols=cols)
    print(f"needed {need.sum():,}  attempted {(need & have).sum():,}  missing {miss.sum():,}  "
          f"coverage {(need & have).sum() / need.sum():.3f}  sessions with missing {len(np.unique(rows))}")


if __name__ == "__main__":
    q.OUT.mkdir(parents=True, exist_ok=True)
    main()
