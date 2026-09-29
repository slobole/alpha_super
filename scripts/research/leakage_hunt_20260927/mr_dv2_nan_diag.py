"""DV2 silent-drop diagnostic (research-only): which PIT members does ``close.unstack().dropna()`` remove, and why?

For every decision date 2004-01-02..2026-08-19 and every S&P 500 PIT member (production trimmed universe) with a
finite Close, count members removed because some field is NaN, broken down by field.  Also checks whether talib
NATR goes permanently NaN after an interior gap (Wilder recursion does not restart after NaN).
Writes mr/dv2_nan_diag.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mr_common as mc
import mr_data


def main() -> None:
    data = mr_data.load("dv2")
    pricing = data["pricing_df"].loc[: mc.STUDY_END]
    strategy = mc.make_dv2(data["universe_trimmed"])
    signal = strategy.compute_signals(pricing)
    universe = data["universe_trimmed"]
    dates = signal.index[(signal.index >= "2004-01-02") & (signal.index <= mc.STUDY_END)]
    fields = signal.columns.get_level_values(1).unique().tolist()
    member = universe.reindex(index=dates, method="ffill").fillna(0).astype(bool)
    symbols = [s for s in member.columns if (s, "Close") in signal.columns]
    member = member[symbols]
    close_ok = signal.xs("Close", axis=1, level=1).reindex(index=dates, columns=symbols).notna()
    base = member & close_ok
    total = int(base.to_numpy().sum())
    by_field = {}
    any_nan = np.zeros(base.shape, dtype=bool)
    for field in fields:
        frame = signal.xs(field, axis=1, level=1).reindex(index=dates, columns=symbols)
        nan = frame.isna().to_numpy() & base.to_numpy()
        by_field[field] = int(nan.sum())
        any_nan |= nan
    dropped = pd.DataFrame(any_nan, index=dates, columns=symbols)
    by_year = dropped.groupby(dropped.index.year).sum().sum(axis=1)
    per_symbol = dropped.sum().sort_values(ascending=False)
    # NATR permanently NaN after an interior gap?
    natr = signal.xs("natr", axis=1, level=1)
    close = signal.xs("Close", axis=1, level=1)
    stuck = []
    for symbol in symbols:
        c, n = close[symbol], natr[symbol]
        valid = c.notna()
        if valid.sum() < 300:
            continue
        first, last = c[valid].index[0], c[valid].index[-1]
        span = c.loc[first:last]
        gap = span.isna().any()
        if gap and n.loc[last:last].isna().all():
            stuck.append(symbol)
    out = {"member_days_with_close": total, "member_days_dropped_by_dropna": int(any_nan.sum()),
           "share_dropped": float(any_nan.sum() / max(total, 1)), "dropped_by_field": by_field,
           "dropped_by_year": {int(k): int(v) for k, v in by_year.items()},
           "top_symbols_dropped_days": {k: int(v) for k, v in per_symbol.head(25).items()},
           "symbols_with_interior_close_gap_and_natr_nan_at_end": stuck[:50], "n_stuck": len(stuck)}
    (mc.OUT / "dv2_nan_diag.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
