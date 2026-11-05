"""The Strategy map: every promoted strategy on one screen, served by Bench at ``/map``.

Three steps, all under ``python -m alpha.bench.engine_map``:

  * ``build``    reads the frozen fund frame and the refreshed sessions and writes
                 ``results/bench/engine_map/data.json`` (and ``build_summary.json``).
                 No backtest runs. Repository checks warn instead of stopping it.
  * ``refresh``  re-runs every engine from 2024-01-02 into ``_work/daily`` (about
                 6 to 9 minutes, 5 workers), then builds.
  * ``trading-facts`` measures how each engine trades from its saved research run,
                 by hand, into ``_work/trading_facts.json``.

The page computes everything else in the browser from daily returns. This module
holds only paths, so importing it runs nothing.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = PACKAGE_DIR_PATH.parents[2]
OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results" / "bench" / "engine_map"
WORK_DIR_PATH = OUTPUT_DIR_PATH / "_work"
DAILY_DIR_PATH = WORK_DIR_PATH / "daily"
DATA_PATH = OUTPUT_DIR_PATH / "data.json"
BUILD_SUMMARY_PATH = OUTPUT_DIR_PATH / "build_summary.json"
MAP_BASE_PATH = PACKAGE_DIR_PATH / "map_base.json"

# What `seed` copies from an older working folder: the inputs a build reads and
# the refreshed sessions. Notes, scratch runs and publish marks stay behind.
SEED_SUFFIX_TUPLE = (".json", ".parquet", ".log")
SEED_SKIP_NAME_TUPLE = ("build_summary.json", "published_data.json")
