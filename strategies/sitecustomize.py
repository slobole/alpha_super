"""
Ensure the repository root is importable when strategy scripts are executed by
file path from the `strategies/` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)

if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
