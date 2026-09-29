"""Control: repeat the sector signal-invariance test after up-casting the REAL Norgate frame to float64.

Norgate delivers float32 prices.  Rescaling a float32 column re-rounds every price at ~6e-8 relative precision, and
IBS = (C-L)/(H-L) amplifies that by C/(H-L) (~100x), so cells within ~1e-5 of the IBS thresholds can flip.  If every
cell matches in float64, the decision logic is exactly scale-invariant and the float32 flips are storage-precision
threshold ties (a real vendor re-adjustment is also re-rounded to float32, so they are real but immaterial).
Usage: python def_sector_f64.py vox_iyr|kie_ihi
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_sector as ds  # noqa: E402
from def_common import FACTORS, cached, harness  # noqa: E402


def main():
    log = harness.ResultLog(f"sector_{ds.POD}_float64_control")
    px = cached(f"sector_{ds.POD}_pricing", ds.load).astype("float64")
    base = ds.sig_tables(px)
    for s in ds.SYMS:
        for k in FACTORS:
            res = ds.compare(base, ds.sig_tables(harness.rescale_symbol_history(px, s, k)))
            log.add("invariance_signal_float64", f"{s}_k{k}", res["passed"], res)
    log.save(f"def_sector_{ds.POD}_float64_control")


if __name__ == "__main__":
    main()
