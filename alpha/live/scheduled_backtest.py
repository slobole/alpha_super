"""Bound scheduled diagnostics to their New York window; never connect to a broker."""

from datetime import datetime, time
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def deadline_ts(now_ts: datetime, *, rulers_only_bool: bool) -> datetime | None:
    market_ts = now_ts.astimezone(MARKET_TIMEZONE_OBJ)
    start_time = time(10) if rulers_only_bool else time(6, 30)
    end_time = time(18) if rulers_only_bool else time(9)
    allowed_day_bool = market_ts.weekday() == 5 if rulers_only_bool else market_ts.weekday() < 5
    if not allowed_day_bool or not start_time <= market_ts.time() < end_time:
        return None
    return datetime.combine(market_ts.date(), end_time, MARKET_TIMEZONE_OBJ)


def stop_process_tree(process_obj) -> None:
    if process_obj.poll() is not None:
        return
    if os.name == "nt":
        # Only the child launched by this diagnostic, including its descendants.
        result_obj = subprocess.run(["taskkill.exe", "/PID", str(process_obj.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        # A venv launcher can exit while taskkill terminates its interpreter child.
        if result_obj.returncode and process_obj.poll() is None:
            result_obj.check_returncode()
    else:
        process_obj.kill()
    process_obj.wait(timeout=10)


def run_scheduled_check(*, rulers_only_bool=False, now_fn=None, launch_fn=None) -> int:
    now_fn = now_fn or (lambda: datetime.now(MARKET_TIMEZONE_OBJ))
    launch_fn = launch_fn or subprocess.Popen
    stop_ts = deadline_ts(now_fn(), rulers_only_bool=rulers_only_bool)
    if stop_ts is None:
        print("[SKIP] Outside the scheduled diagnostic's New York window.")
        return 0
    command_list = [sys.executable, "-m", "alpha.live.backtest_check"]
    if rulers_only_bool:
        command_list.append("--rulers-only")
    process_obj = launch_fn(command_list)
    try:
        while process_obj.poll() is None:
            remaining_float = (stop_ts - now_fn().astimezone(MARKET_TIMEZONE_OBJ)).total_seconds()
            if remaining_float <= 0:
                print("[FAIL] Diagnostic stopped at its absolute New York deadline.")
                return 1
            try:
                return process_obj.wait(timeout=min(remaining_float, 1.0))
            except subprocess.TimeoutExpired:
                pass
        return process_obj.returncode
    finally:
        stop_process_tree(process_obj)


if __name__ == "__main__":
    import argparse

    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--rulers-only", action="store_true")
    raise SystemExit(run_scheduled_check(rulers_only_bool=parser_obj.parse_args().rulers_only))
