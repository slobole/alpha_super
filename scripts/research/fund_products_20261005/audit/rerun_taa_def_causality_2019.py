"""Second end-date causality check: runs ending 2019-12-31 (a month-end session) vs the run ending 2026-10-02.

A causal strategy gives the same NAV, invested value, cash and fills on every date up to the earlier end date,
whatever the later end date is. Read-only except for one JSON under the audit output folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rerun_taa_def_compare import ENGINE_ALIAS_TUPLE, NEW_DIR_PATH, compare_paths, compare_tx, read_path, read_tx

EARLY_DIR_PATH = NEW_DIR_PATH / "end_20191231"
EARLY_END_TS = pd.Timestamp("2019-12-31")


def main() -> None:
    out = {}
    for alias in ENGINE_ALIAS_TUPLE:
        early, late = read_path(EARLY_DIR_PATH, alias), read_path(NEW_DIR_PATH, alias)
        path_cmp = compare_paths(early, late, EARLY_END_TS)
        tx_cmp = compare_tx(read_tx(EARLY_DIR_PATH, alias), read_tx(NEW_DIR_PATH, alias), EARLY_END_TS)
        out[alias] = {"path": path_cmp, "tx": {k: v for k, v in tx_cmp.items() if k != "first_diverging_rows"},
                      "tx_first_diverging_rows": tx_cmp.get("first_diverging_rows", [])}
        print(alias, "rows", path_cmp["a_rows"], path_cmp["b_rows"], "same_index", path_cmp["same_index"],
              "max_nav_rel_diff", path_cmp["max_abs_nav_rel_diff"], "max_ret_diff",
              path_cmp["max_abs_daily_return_diff"], "path_identical", path_cmp["identical_within_tol"],
              "| tx", tx_cmp["a_count"], tx_cmp["b_count"], "tx_identical", tx_cmp["identical_within_tol"])
    (NEW_DIR_PATH / "compare" / "causality_end_20191231.json").write_text(json.dumps(out, indent=2, default=str),
                                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
