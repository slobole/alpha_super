"""defensive_delta audit: how much do the DV2-IND slots lean on the pre-2010 research-run fill (frame s4 = idle 0% before 2010)?
Also prints the frame facts the orchestrator needs (window, days, columns). Read-only; writes fill_sensitivity.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import data.norgate_loader  # noqa: F401
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20260930"))
import numpy as np, pandas as pd
import fp_lib as fp
import a6
from a6 import Lab
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/defensive_delta"
lab = Lab()
stored = json.loads((fp.STUDY / "report" / "a6d.json").read_text(encoding="utf-8"))
res = {"frame": {"start": str(lab.start.date()), "end": str(fp.END.date()), "n_days": int(len(lab.ret(a6.GROWTH))),
                 "columns": list(lab.frame.columns), "fingerprint": lab.fp}, "slots": {}}
fr_main, fr_s4 = lab.frames["main"][0], lab.frames["s4_etf_idle_pre2010"][0]
etf_first = lab.data["sleeve"]["etf_dv2"].first_valid_index()
fill = fr_main.loc[lab.start:etf_first, "etf_dv2"].iloc[:-1]
res["etf_dv2_fill"] = {"engine_first_return": str(etf_first.date()), "fill_days": int(len(fill)),
                       "fill_cum_return": float(np.prod(1 + fill.to_numpy()) - 1), "fill_nonzero_days": int((fill != 0).sum()),
                       "fill_max_dd": float((np.r_[1, np.cumprod(1 + fill.to_numpy())] / np.maximum.accumulate(np.r_[1, np.cumprod(1 + fill.to_numpy())]) - 1).min())}
print(res["frame"]["start"], res["frame"]["end"], res["frame"]["n_days"], "| fill", res["etf_dv2_fill"])
for slot in ("next", "calm_next", "target"):
    v = stored["defensive"][slot]
    rows = {}
    for fk in ("main", "s4_etf_idle_pre2010"):
        r = lab.ret(v["weights"], 1.0, fk)
        nav = np.r_[1, np.cumprod(1 + r.to_numpy())]
        rf = lab.frames[fk][0][fp.TBILL].reindex(r.index).to_numpy()
        x = r.to_numpy() - rf
        gfc = float(a6.lib.common.window_return_float(r, *a6.lib.CRISIS_DICT["gfc"]))
        rows[fk] = {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "xs": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
                    "dd": float((nav / np.maximum.accumulate(nav) - 1).min()), "gfc": gfc}
    res["slots"][slot] = {"name": v["name"], **rows}
    print(slot, v["name"], {k: {m: round(x, 4) for m, x in d.items()} for k, d in rows.items()})
(OUT / "fill_sensitivity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
