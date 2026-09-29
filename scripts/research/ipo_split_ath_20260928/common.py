"""Shared paths, constants and helpers for the IPO / split all-time-high study (research only).

Frozen plan: docs/research/IPO_SPLIT_ATH_PREREG_20260928.md.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
SCRIPTS_RESEARCH_PATH = REPO_ROOT_PATH / "scripts" / "research"
for path_obj in (REPO_ROOT_PATH, SCRIPTS_RESEARCH_PATH):
    if str(path_obj) not in sys.path:
        sys.path.insert(0, str(path_obj))

MAIN_CHECKOUT_PATH = Path(r"C:/Users/User/Documents/workspace/alpha_super")
RESULTS_DIR_PATH = Path(os.environ.get("IPO_SPLIT_ATH_OUT", str(MAIN_CHECKOUT_PATH / "results" / "research" / "ipo_split_ath_20260928")))
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"
CHART_DIR_PATH = RESULTS_DIR_PATH / "charts"


def log_progress(message_str: str) -> None:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    stamp_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_DIR_PATH / "progress_log.txt", "a", encoding="utf-8") as file_obj:
        file_obj.write(f"{stamp_str} {message_str}\n")
    print(f"[{stamp_str}] {message_str}", flush=True)


def write_json(name_str: str, payload_obj) -> Path:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj = RESULTS_DIR_PATH / name_str
    path_obj.write_text(json.dumps(payload_obj, indent=1, default=str), encoding="utf-8")
    return path_obj
