"""A1: parameter plateau sweep on the development sample (1970-2022)."""
from common import *
import itertools, time
panel = dev(load())
grid = dict(vol_window=[21, 42, 63, 126, 252], target_vol=[0.06, 0.07, 0.08, 0.09, 0.10],
            trigger_buffer=[0.0, 0.005, 0.01], rebalance=["W", "M", "Q"], band=[0.0, 0.05])
rows = []
t0 = time.time()
for i, (vw, tv, tb, rb, bd) in enumerate(itertools.product(*grid.values())):
    p = Params(vol_window=vw, port_vol_window=vw, target_vol=tv, trigger_buffer=tb, rebalance=rb, band=bd)
    res = run(panel, p)
    row = {"vol_window": vw, "target_vol": tv, "trigger_buffer": tb, "rebalance": rb, "band": bd}
    row.update(summarize(res))
    rows.append(row)
    if i % 50 == 0:
        print(i, round(time.time() - t0), flush=True)
df = pd.DataFrame(rows)
save(df, "plateau_grid", "a1")
print("done", len(df), round(time.time() - t0))
