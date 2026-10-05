"""Timing-lens reviewer: load the study Lab once and cache frames / inputs to a pickle for the other tl_* scripts."""
import pickle, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
t0 = time.time()
import g_lib as g
lab = g.Lab()
OUT = g.STUDY / "audit" / "review" / "timing_lens"
OUT.mkdir(parents=True, exist_ok=True)
d = lab.data
keep = {k: d[k] for k in ("sleeve", "long", "long_unscaled", "long_etf_cash", "stressed_long", "stressed_exact", "cash_long", "cash_exact",
                          "bench", "index", "dtb3", "path", "full", "mr_cash_run", "mr_cash_add", "drag_ex_bil", "drag", "tx", "nav", "meta")}
pickle.dump({"data": keep, "frames": lab.frames}, open(OUT / "cache.pkl", "wb"))
print("sys.path[0:4]", sys.path[:4])
import data.norgate_loader as nl, strategies.mr_capsule.vix_stress_gate as vg
print("norgate_loader from", nl.__file__)
print("vix gate from", vg.__file__)
import common, evaluation
print("common from", common.__file__, "evaluation from", evaluation.__file__)
print("lib from", g.lib.__file__, "ga from", g.ga.__file__)
print("frames", {k: (v[0].shape, str(v[1].date())) for k, v in lab.frames.items()})
print("done", round(time.time() - t0, 1))
