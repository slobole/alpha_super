"""Would MAIN's uncommitted capsule edits change a backtest? Empirical preconditions (read-only).

Edit 1 (parking ETFs exempt from missing-price liquidation) matters only if BIL has no bar on a pricing session while held.
Edit 2 (gate switch read against the previous PRICING session, not the previous VIX row) matters only if the VIX
calendar and the pricing calendar differ around a gate switch.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
from data.norgate_loader import load_raw_prices
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser, stress_gate_open_ser, gate_state_at

nav = pd.read_csv(WT / "results/research/mr_capsule_build_20261004/dv2_bil_nav.csv", index_col=0, parse_dates=True)
cal = nav.index
px = load_raw_prices(["BIL"], ["$SPX"], start_date="2004-01-01")
bil = px[("BIL", "Close")]
bil_open = px[("BIL", "Open")]
first = bil.dropna().index[0]
c2 = cal[cal >= first]
print("BIL first bar", first.date(), "pricing sessions since", len(c2), "BIL close missing", int(bil.reindex(c2).isna().sum()), "BIL open missing", int(bil_open.reindex(c2).isna().sum()))
print("missing dates", [str(d.date()) for d in c2[bil.reindex(c2).isna()]][:20])
vix = load_vix_close_ser()
gate = stress_gate_open_ser(vix)
vcal = vix.index[vix.index >= cal[0]]
print("VIX sessions since 2004-01-02", len(vcal), "pricing sessions", len(cal))
print("pricing sessions without VIX", [str(d.date()) for d in cal.difference(vcal)][:20])
print("VIX sessions not in pricing calendar", [str(d.date()) for d in vcal.difference(cal)][:20])
# old vs new gate-switch definition over all decision closes
diff = []
for i in range(1, len(cal)):
    d = cal[i]
    g = gate_state_at(gate, d)
    pos = int(gate.index.searchsorted(d, side="right")) - 1
    old = False if pos < 1 else bool(gate.iloc[pos - 1]) != bool(g)
    new = gate_state_at(gate, cal[i - 1]) != bool(g)
    if old != new:
        diff.append(str(d.date()))
print("decision closes where HEAD and MAIN-edit gate-switch flags differ:", len(diff), diff[:10])
sw = gate.reindex(cal).astype(int).diff().abs().fillna(0)
print("gate switches 2004-on:", int(sw.sum()), "open share", float(gate.reindex(cal).mean()))
print("last gate rows:\n", pd.DataFrame({"vix": vix, "gate": gate}).tail(8).to_string())
