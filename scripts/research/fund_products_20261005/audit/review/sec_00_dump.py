"""Reviewer scratch (secondary lens): dump the study's frames and raw sleeve files to one pickle so that the checks
can be done with plain pandas / numpy, without the study's book functions."""
import pickle
import sys
from pathlib import Path

import pandas as pd

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20261005"))
import g_lib as g  # noqa: E402
from g_lib import Lab, lib  # noqa: E402

OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
OUT.mkdir(parents=True, exist_ok=True)
lab = Lab()
d = lab.data
keep_frames = ["main", "s1_house_cash", "s3_plus_5bps", "s6_exact"]
aliases = ["taa3x", "taa3x_1n", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g", "ndx_vxn", "core5", "btal_qqq", "dv2", "hpi_vote"]
dump = {
    "frames": {k: lab.frames[k] for k in keep_frames},
    "tx": {a: d["tx"][a] for a in aliases},
    "nav": {a: d["nav"][a] for a in aliases},
    "path_new": {a: d["path"][a] for a in d["path"]},
    "path_old": {a: lib.read_path(lib.SOURCE, a) for a in ["taa3x", "taa3x_1n", "ndx_vxn", "core5", "btal_qqq"]},
    "bench": d["bench"],
    "dtb3": d["dtb3"],
    "sleeve": d["sleeve"],
    "index": d["index"],
    "crisis": lib.CRISIS_DICT,
    "source_dir": str(lib.SOURCE),
}
with open(OUT / "dump.pkl", "wb") as h:
    pickle.dump(dump, h)
print("frames", {k: (v[0].shape, str(v[1].date())) for k, v in dump["frames"].items()})
print("main cols", list(dump["frames"]["main"][0].columns))
print("bench cols", list(d["bench"].columns))
print("tx cols", list(d["tx"]["taa3x"].columns))
print("path cols", list(dump["path_old"]["taa3x"].columns), list(dump["path_new"]["dv2_g"].columns))
print("source", lib.SOURCE)
