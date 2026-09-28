"""A3 detail: which fields of load_exact_hpi_inputs(end_date_str=T) differ from the full load truncated at T.

Usage: uv run python hpi_loader_prefix_diag.py
"""

from __future__ import annotations

import pandas as pd

import hpi_common as hc
from strategies.hpi import stateful_long as hpi_mod

CUTS = ("2008-10-10", "2020-03-16", "2025-06-13", "2026-08-31")


def main() -> None:
    full = hc.load_full_inputs()["pricing_df"]
    rows = []
    for cut_str in CUTS:
        cut = pd.Timestamp(cut_str)
        _, _, p = hpi_mod.load_exact_hpi_inputs("S&P 500", hc.BENCH, "1998-01-01", cut_str)
        t = full.loc[:cut]
        cols = t.columns.intersection(p.columns)
        a, b = p[cols], t[cols]
        ne = ~((a == b) | (a.isna() & b.isna()))
        bad_cols = ne.any()[ne.any()].index
        fields = sorted({c[1] for c in bad_cols})
        bad_rows = sorted({str(d.date()) for c in bad_cols for d in ne.index[ne[c].to_numpy()]})
        rows.append({"cut": cut_str, "n_cols_differing": int(len(bad_cols)), "fields_differing": fields,
                     "dates_differing": bad_rows, "last_two_sessions": [str(d.date()) for d in t.index[-2:]],
                     "example": [(c[0], c[1], a.loc[t.index[-2]:, c].tolist(), b.loc[t.index[-2]:, c].tolist())
                                 for c in list(bad_cols)[:3]]})
        print(rows[-1])
    hc.dump_json(rows, "invariance/loader_prefix_diag.json")


if __name__ == "__main__":
    main()
