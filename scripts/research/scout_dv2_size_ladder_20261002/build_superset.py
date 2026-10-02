"""Build the size-ladder superset panel: every symbol that was a member of any ladder index between 2001-01-01 and the
vault seal (2022-12-30), with one exact membership matrix per index (alpha.scout.universes).

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/build_superset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from register import UNIVERSE_TUPLE

from alpha.scout.universes import build_superset_panel

SUPERSET_NAME_STR = "size_ladder"


def main() -> None:
    started_float = time.time()
    folder_path = build_superset_panel([name_str for name_str, _ in UNIVERSE_TUPLE], SUPERSET_NAME_STR,
                                       log_fn=lambda s: print(f"[{time.time() - started_float:6.0f}s] {s}", flush=True))
    print("built", folder_path, flush=True)


if __name__ == "__main__":
    main()
