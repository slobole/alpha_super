"""Read-only: is each sleeve source's strategy module (2026-09-29 runs, MAIN shelf_rebuild sources) unchanged at main HEAD?

The recorded module_sha256_str was taken on MAIN's working file (mixed line endings), so the test is:
recorded hash == sha256(MAIN working file)  AND  MAIN file == WT (HEAD) file after stripping carriage returns.
Only the strategy module file itself is compared; shared base classes / engine code are NOT covered.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
S = MAIN / "results/research/portfolio/shelf_rebuild_20260929/sources"
USED = {"core5", "btal_qqq", "taa3x", "taa3x_1n", "ndx_vxn", "etf_dv2", "eom_flow", "downshock", "dv2", "hpi_vote", "hpi_ibs_rsi", "ndx_atr", "ndx_natr20"}
norm = lambda b: hashlib.sha256(b.replace(b"\r", b"")).hexdigest()  # noqa: E731
out = []
for f in sorted(glob.glob(str(S / "*__metadata.json"))):
    m = json.load(open(f, encoding="utf-8"))
    if m["alias_str"] not in USED:
        continue
    wt, mn = WT / m["module_path_str"], MAIN / m["module_path_str"]
    rec_is_main = hashlib.sha256(mn.read_bytes()).hexdigest() == m["module_sha256_str"]
    same_content = norm(wt.read_bytes()) == norm(mn.read_bytes())
    out.append(f"{m['alias_str']:12s} | recorded==MAIN file: {rec_is_main} | MAIN==HEAD (CR stripped): {same_content} | {m['module_path_str']}")
print("\n".join(out))
(WT / "results/research/portfolio/fund_products_20261005/audit/report_map/module_hash.txt").write_text("\n".join(out), encoding="utf-8")
