from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "live_events.jsonl")


def log_event(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> None:
    log_path_obj = Path(log_path_str)
    log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    event_record_dict = {
        "event_name_str": event_name_str,
        "event_timestamp_str": datetime.now(timezone.utc).isoformat(),
        **event_payload_dict,
    }
    with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
        log_file_obj.write(json.dumps(event_record_dict, sort_keys=True) + "\n")
