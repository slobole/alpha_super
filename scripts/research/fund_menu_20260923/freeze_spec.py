"""Freeze frozen_spec.yaml: write its sha256 and log the freeze before any book is built.

build_books.py refuses to run if the spec bytes differ from this hash, and the
experiment ledger records that no book result existed at freeze time.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent / "frozen_spec.yaml"
HASH_PATH = Path(__file__).resolve().parent / "frozen_spec.sha256"


def main() -> int:
    book_dir_path = common.STUDY_DIR_PATH / "books"
    existing_book_list = sorted(p.name for p in book_dir_path.glob("*")) if book_dir_path.exists() else []
    if existing_book_list:
        raise RuntimeError(f"Book outputs already exist ({existing_book_list[:3]}...); a freeze after results is not a freeze.")
    spec_sha_str = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    HASH_PATH.write_text(f"{spec_sha_str}  frozen_spec.yaml\n", encoding="utf-8")
    event_dict = {
        "event_str": "spec_frozen",
        "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_sha256_str": spec_sha_str,
        "book_outputs_present_bool": False,
        "inputs_seen_before_freeze_list": [
            "inventory/sleeve_stats_full.csv", "inventory/sleeve_stats_common.csv", "inventory/sleeve_activity.csv",
            "inventory/corr_*.csv", "inventory/correlation_clusters.csv", "construction/feasibility.csv",
        ],
        "note_str": "Only sleeve-level facts and risk (volatility, correlation, cadence) were inspected; no book was evaluated.",
    }
    with (common.STUDY_DIR_PATH / "experiment_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(json.dumps(event_dict, sort_keys=True) + "\n")
    print(json.dumps(event_dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
