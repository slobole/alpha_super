"""Re-check the CORE5 DBC decision-table invariance with the decision tolerance (state/flag exact, weight 1e-6),
and the same cases after up-casting to float64 (separates float32 storage / rolling-std noise from logic)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_core5 as c
from def_common import FACTORS, harness

log = harness.ResultLog("core5_recheck")
px = c.load_pricing()
for dtype in ("float32", "float64"):
    p = px.astype(dtype)
    base = c.decision_table(p)
    for k in FACTORS:
        for case, ns in (("split", ["DBC", c.NS("DBC")]), ("tr_dividend_only", [c.NS("DBC")])):
            res = c.compare_tables(base, c.decision_table(c.rescale_frame(p, ns, k)))
            log.add(f"invariance_decision_table_{dtype}", f"DBC_{case}_k{k}", res["passed"], res)
log.save("def_core5_recheck")
