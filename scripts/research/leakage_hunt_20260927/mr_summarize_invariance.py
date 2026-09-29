"""Collect mr/invariance/*.json into one table (mr/invariance_summary.csv) and print it."""

from __future__ import annotations

import json

import pandas as pd

import mr_common as mc


def main() -> None:
    rows = []
    for path in sorted((mc.OUT / "invariance").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            d = row["detail"]
            rows.append({"file": path.stem, "strategy": row["strategy"], "test": row["test"], "case": row["case"],
                         "passed": row["passed"], "n_ref": d.get("n_decisions_reference"),
                         "n_cand": d.get("n_decisions_candidate"), "only_ref": d.get("n_only_reference"),
                         "only_cand": d.get("n_only_candidate"), "first_div": d.get("first_divergence_date"),
                         "max_rel_notional": d.get("max_rel_notional_diff"),
                         "n_notional_beyond_2pct": d.get("n_notional_beyond_tol"),
                         "ex_ref": d.get("examples_only_reference"), "ex_cand": d.get("examples_only_candidate"),
                         "error": d.get("error")})
    frame = pd.DataFrame(rows)
    frame.to_csv(mc.OUT / "invariance_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(frame.drop(columns=["ex_ref", "ex_cand"]).to_string(index=False))
    print(frame.groupby(["strategy", "test"])["passed"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
