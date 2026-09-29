"""(d)/(e) Full VANILLA engine runs of both NDX modules from cached data (trimmed = as committed; untrimmed =
membership without the loader's iloc[:-5] tail trim).  One process per (module, variant) so they can run in
parallel.

Usage: uv run python scripts/research/leakage_hunt_20260927/ndx_backtests.py atr_vxn trimmed
"""

from __future__ import annotations

import sys

import ndx_common as nc


def main() -> None:
    key, variant = sys.argv[1], sys.argv[2]
    data = nc.load_data(variant)
    strategy = nc.run_backtest(key, data)
    path = nc.save_run(strategy, f"{key}_{variant}")
    print("saved", path)


if __name__ == "__main__":
    main()
