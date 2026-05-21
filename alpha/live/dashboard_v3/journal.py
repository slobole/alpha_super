"""Operator intervention journal (append-only JSONL).

Every confirmed dashboard action appends one row here. Operators can
revisit "did I touch this last night?" without trawling event logs.
The file is intentionally human-friendly newline-delimited JSON so
``grep``/``tail`` work on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_JOURNAL_PATH_STR = "alpha/live/logs/operator_journal.jsonl"


@dataclass
class JournalEntry:
    timestamp_str: str
    pod_id_str: str
    mode_str: str
    action_name_str: str
    job_id_str: str
    initial_status_str: str
    actor_str: str = "operator"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_str": self.timestamp_str,
            "pod_id_str": self.pod_id_str,
            "mode_str": self.mode_str,
            "action_name_str": self.action_name_str,
            "job_id_str": self.job_id_str,
            "initial_status_str": self.initial_status_str,
            "actor_str": self.actor_str,
        }


def append_journal_entry(
    pod_id_str: str,
    mode_str: str,
    action_name_str: str,
    job_id_str: str,
    initial_status_str: str,
    *,
    actor_str: str = "operator",
    journal_path_str: str = DEFAULT_JOURNAL_PATH_STR,
) -> JournalEntry:
    entry_obj = JournalEntry(
        timestamp_str=datetime.now(timezone.utc).isoformat(),
        pod_id_str=pod_id_str,
        mode_str=mode_str,
        action_name_str=action_name_str,
        job_id_str=job_id_str,
        initial_status_str=initial_status_str,
        actor_str=actor_str,
    )
    journal_path_obj = Path(journal_path_str)
    journal_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with journal_path_obj.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(entry_obj.as_dict(), separators=(",", ":")) + "\n")
    return entry_obj


def read_journal_entry_dict_list(
    *,
    limit_int: int = 200,
    journal_path_str: str = DEFAULT_JOURNAL_PATH_STR,
) -> list[dict[str, Any]]:
    journal_path_obj = Path(journal_path_str)
    if not journal_path_obj.exists():
        return []
    entry_dict_list: list[dict[str, Any]] = []
    with journal_path_obj.open("r", encoding="utf-8") as file_obj:
        for line_str in file_obj:
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                entry_dict = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            entry_dict_list.append(entry_dict)
    return entry_dict_list[-limit_int:]
