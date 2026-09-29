"""Tally pass/fail of every defensive-audit test JSON and copy the JSONs into results/.../def/."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import OUT

import pandas as pd

ROOT = OUT.parent
rows = []
for path in sorted(ROOT.glob("def_*__invariance.json")):
    shutil.copy2(path, OUT / path.name)
    for rec in json.loads(path.read_text(encoding="utf-8")):
        rows.append({"file": path.name, "strategy": rec["strategy"], "test": rec["test"], "passed": rec["passed"]})
df = pd.DataFrame(rows)
tab = df.groupby(["file", "strategy", "test"])["passed"].agg(n="count", n_pass="sum").reset_index()
tab["n_fail"] = tab["n"] - tab["n_pass"]
tab.to_csv(OUT / "def_test_summary.csv", index=False)
print(tab.to_string())
