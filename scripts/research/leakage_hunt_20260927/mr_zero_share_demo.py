"""Targeted invariance case for the adjusted-unit zero-share cancellation (research-only).

A future 1:500 reverse split (k = 0.002) multiplies a name's whole adjusted history by 500, the same thing Norgate's
CAPITALSPECIAL already did to WFRD (2008 adjusted Close $62,855 vs raw $43.58) and CHKAQ.  With production
accounting int(slot_value / adjusted_close) becomes 0 and the engine silently cancels the entry, so a FUTURE corporate
action changes a PAST decision.  historical_share_units_bool=True sizes from Unadjusted Close and must be invariant.
Writes mr/zero_share_demo_<family>.json.
Usage: uv run python mr_zero_share_demo.py <dv2|hpi>
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import harness
import mr_common as mc
import mr_data
from mr_invariance import build, decisions, passed

START, END, K = "2024-01-02", "2024-12-31", 0.002


def main(family: str) -> None:
    data = mr_data.load(family)
    universe = data["universe_trimmed"] if family == "dv2" else data["universe"]
    pricing = mc.subset_pricing(data["pricing_df"].loc[:END], universe, START, END,
                                bench=("$SPX",) if family == "dv2" else ("$SPXTR",))
    rows = []
    for hsu in (False, True):
        ref = decisions(mc.run(build(family, data, hsu=hsu), pricing, START, END))
        top = ref[ref["side"] > 0]["asset"].value_counts().index[:3].tolist()
        cand_pricing = pricing
        for symbol in top:
            cand_pricing = harness.rescale_symbol_history(cand_pricing, symbol, K)
        cand_pricing.attrs.update(pricing.attrs)
        cmp = harness.compare_transaction_decisions(ref, decisions(mc.run(build(family, data, hsu=hsu), cand_pricing,
                                                                           START, END)), END)
        cmp["rescaled_symbols"] = top
        rows.append({"test": f"invariance_{'hsu' if hsu else 'engine'}_k{K}", "passed": passed(cmp), "detail": cmp})
        print(rows[-1]["test"], rows[-1]["passed"], {k: cmp[k] for k in ("n_only_reference", "n_only_candidate")})
    (mc.OUT / f"zero_share_demo_{family}.json").write_text(json.dumps(rows, indent=2, default=str),
                                                                        encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
