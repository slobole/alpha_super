"""CLI: ``python -m alpha.bench.engine_map build | refresh | trading-facts | seed --from DIR``.

``refresh`` runs every engine, then builds, as the nightly job did: if the run
died or the QQQ/T-bill file failed, it stops before the build so the page keeps
its last good data. Engines that failed keep their last good series; the build
still runs and the exit code is 1 so the Bench job reads failed.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

from alpha.bench.engine_map import (
    DAILY_DIR_PATH,
    SEED_SKIP_NAME_TUPLE,
    SEED_SUFFIX_TUPLE,
    WORK_DIR_PATH,
)


def run_build() -> int:
    """Run build.py once. It is a script: importing it is the build."""
    runpy.run_module("alpha.bench.engine_map.build", run_name="__main__")
    return 0


def run_refresh(refresh_arg_list: list[str]) -> int:
    from alpha.bench.engine_map import refresh

    started_str = datetime.now().isoformat(timespec="seconds")
    refresh_code_int = refresh.main(refresh_arg_list)
    if "--check" in refresh_arg_list:  # a read-only look at the runs on disk: nothing to build
        return refresh_code_int
    status_path = WORK_DIR_PATH / "refresh_status.json"
    try:
        status_dict = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status_dict = {}
    # refresh writes "finished" last, so an older one means it died first.
    if status_dict.get("finished", "") < started_str:
        print("refresh did not finish: the map keeps its last build", flush=True)
        return 1
    if status_dict.get("bench_error"):
        print(f"QQQ and T-bills not refreshed ({status_dict['bench_error']}): the map keeps its last build", flush=True)
        return 1
    run_build()
    return 1 if refresh_code_int else 0


def run_seed(source_path: Path) -> int:
    """Copy an older working folder's inputs and refreshed sessions into _work."""
    if not source_path.is_dir():
        print(f"no such folder: {source_path}")
        return 1
    copied_int = 0
    for sub_str in ("", "daily"):
        source_dir_path = source_path / sub_str if sub_str else source_path
        target_dir_path = DAILY_DIR_PATH if sub_str else WORK_DIR_PATH
        if not source_dir_path.is_dir():
            continue
        target_dir_path.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(source_dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix in SEED_SUFFIX_TUPLE and file_path.name not in SEED_SKIP_NAME_TUPLE:
                shutil.copy2(file_path, target_dir_path / file_path.name)
                copied_int += 1
    print(f"copied {copied_int} files into {WORK_DIR_PATH}")
    return 0


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(prog="python -m alpha.bench.engine_map", description=__doc__)
    sub_obj = parser_obj.add_subparsers(dest="command_str", required=True)
    sub_obj.add_parser("build", help="write data.json from the saved frame and refreshed sessions")
    sub_obj.add_parser("refresh", help="re-run every engine, then build; other arguments go to the refresh")
    sub_obj.add_parser("trading-facts", help="measure how each engine trades, from saved runs")
    seed_obj = sub_obj.add_parser("seed", help="copy an older working folder into _work")
    seed_obj.add_argument("--from", dest="source_str", required=True)
    parsed_obj, rest_list = parser_obj.parse_known_args(argv_list)
    if rest_list and parsed_obj.command_str != "refresh":
        parser_obj.error(f"unrecognized arguments: {' '.join(rest_list)}")
    if parsed_obj.command_str == "build":
        return run_build()
    if parsed_obj.command_str == "refresh":
        return run_refresh(rest_list)
    if parsed_obj.command_str == "trading-facts":
        from alpha.bench.engine_map import trading_facts

        return trading_facts.main()
    return run_seed(Path(parsed_obj.source_str))


if __name__ == "__main__":
    sys.exit(main())
