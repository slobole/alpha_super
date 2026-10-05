"""defensive_delta audit, step 2b: full re-run of a6.py / a6d.py with an EMPTY tail cache, redirected to the audit folder.

The frozen scripts are imported unchanged; only fp_lib.STUDY is re-pointed so that the json, the tail cache and the
ledger line land under results/research/portfolio/fund_products_20261005/audit/defensive_delta/rerun_<which>/ and the
2026-09-30 study folder is not touched. Inputs are read from the main checkout (read-only, no bytecode written).

Usage: python dd_rerun.py a6d | a6
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import data.norgate_loader  # noqa: F401,E402  (bind the `data` package to this worktree's HEAD copy first)

WT = Path(__file__).resolve().parents[4]
FP_DIR = WT / "scripts" / "research" / "fund_products_20260930"
sys.path.insert(0, str(FP_DIR))

import fp_lib as fp  # noqa: E402

which = sys.argv[1]
OLD = fp.STUDY
NEW = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "defensive_delta" / f"rerun_{which}"
(NEW / "report").mkdir(parents=True, exist_ok=True)
if which == "a6":
    shutil.copy(OLD / "report" / "clean.json", NEW / "report" / "clean.json")   # a6's start-up tail check reads it
fp.STUDY = NEW

import a6  # noqa: E402
import a6d  # noqa: E402

assert not (NEW / "report" / "a6_tail_cache.json").exists() or "--keep-cache" in sys.argv, "start from an empty cache"
raise SystemExit({"a6": a6, "a6d": a6d}[which].main())
