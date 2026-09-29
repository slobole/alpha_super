"""Why do engine-unit and historical-share-unit DV2 runs make different decisions? (research-only)

Runs DV2 on one window twice (production accounting vs historical_share_units_bool=True) and prints the first
divergent (date, asset, side) rows with the orders/positions context.  Writes mr/hsu_diff_<window>.json.
Usage: uv run python mr_hsu_diff.py <dv2|hpi> <start> <end>
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import harness
import mr_common as mc
import mr_data


def main(family: str, start: str, end: str) -> None:
    data = mr_data.load(family)
    universe = data["universe_trimmed"] if family == "dv2" else data["universe"]
    pricing = mc.subset_pricing(data["pricing_df"].loc[:end], universe, start, end,
                                bench=("$SPX",) if family == "dv2" else ("$SPXTR",))
    runs = {}
    for hsu in (False, True):
        strategy = mc.make_dv2(universe) if family == "dv2" else mc.make_hpi(universe)
        strategy.historical_share_units_bool = hsu
        mc.run(strategy, pricing, start, end)
        runs[hsu] = strategy
    d0 = harness.transaction_decisions(runs[False].get_transactions())
    d1 = harness.transaction_decisions(runs[True].get_transactions())
    cmp = harness.compare_transaction_decisions(d0, d1, end)
    merged = d0.merge(d1, on=["date", "asset", "side"], how="outer", indicator=True, suffixes=("_eng", "_hsu"))
    diff = merged[merged["_merge"] != "both"].sort_values("date")
    first = diff["date"].min() if len(diff) else None
    context = {}
    if first is not None:
        for hsu, strategy in runs.items():
            tx = strategy.get_transactions()
            window = tx[(tx["bar"] >= first - pd.Timedelta(days=10)) & (tx["bar"] <= first + pd.Timedelta(days=3))]
            context["hsu" if hsu else "engine"] = window.astype(str).to_dict("records")
            out = strategy._captured_stdout.splitlines()
            context[f"stdout_{'hsu' if hsu else 'engine'}"] = [l for l in out if "zero" in l.lower() or "cancel" in l.lower()][:20]
    result = {"compare": cmp, "first_divergence": None if first is None else str(first.date()),
              "diff_rows": diff.astype(str).head(40).to_dict("records"), "context": context}
    (mc.OUT / f"hsu_diff_{family}_{start[:4]}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "context"}, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
