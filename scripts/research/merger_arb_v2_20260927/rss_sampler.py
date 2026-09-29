"""Sample the resident memory of the study's python processes every few seconds (memory budget evidence only).
Usage: python rss_sampler.py <log path> [interval seconds]. Stops when no study process has run for 10 minutes."""

from __future__ import annotations

import sys
import time
from datetime import datetime

import psutil

log_path = sys.argv[1]
interval_float = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
peak_mb_float = 0.0
last_seen_float = time.time()
with open(log_path, "a", encoding="utf-8") as fh:
    while True:
        total_mb_float, label_str = 0.0, ""
        for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
            try:
                cmd_str = " ".join(proc.info["cmdline"] or [])
                if "merger_arb_v2_20260927" in cmd_str and "run_study" in cmd_str:
                    mb_float = proc.info["memory_info"].rss / 1e6
                    total_mb_float += mb_float
                    label_str = cmd_str.split("run_study.py")[-1].strip()[:40]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if total_mb_float > 0:
            last_seen_float = time.time()
            peak_mb_float = max(peak_mb_float, total_mb_float)
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} rss_mb {total_mb_float:.0f} peak_mb {peak_mb_float:.0f} stage {label_str}\n")
            fh.flush()
        elif time.time() - last_seen_float > 600:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} sampler stop, peak_mb {peak_mb_float:.0f}\n")
            break
        time.sleep(interval_float)
