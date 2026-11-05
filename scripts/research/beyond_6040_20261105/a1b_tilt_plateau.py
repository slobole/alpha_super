"""A1b: plateau for the tilted candidate (vol window x target x band), monthly rebalance, SMA10 k=0.5."""
from common import *
import itertools
panel = dev(load())
rows = []
for vw, tv, bd, k in itertools.product([42, 63, 126], [0.06, 0.07, 0.08, 0.09, 0.10], [0.0, 0.05], [0.25, 0.5]):
    p = Params(vol_window=vw, port_vol_window=vw, target_vol=tv, band=bd, tilt="sma10", tilt_strength=k, tilt_freq="M")
    rows.append({"vol_window": vw, "target_vol": tv, "band": bd, "k": k} | summarize(run(panel, p)))
df = pd.DataFrame(rows); save(df, "tilt_plateau_grid", "a1")
print(df[["vol_window","target_vol","band","k","cagr","vol","sharpe","max_dd","worst_12m","turnover_per_year","sh_1970s","sh_1980s","sh_1990s","sh_2000s","sh_2010s","sh_2020-22"]].round(3).to_string())
