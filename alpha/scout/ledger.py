"""Append-only, hash-chained research ledger.

Every registration, trial, station verdict, vault opening and gate result is one
JSON line in `research_ledger/scout_ledger.jsonl`. The file is committed to git.

Chain rule (tamper evidence):

    row_hash_str = sha256( canonical_json(row without "row_hash_str") )
    row["prev_row_hash_str"] == previous row's row_hash_str   (genesis: 64 zeros)
    row["row_id_int"] == previous row_id_int + 1               (genesis: 1)

The hash is taken over the parsed row in canonical form (sorted keys, no
whitespace, UTF-8), not over raw file bytes, so a CRLF checkout does not break
the chain. Editing, deleting, inserting or reordering any row breaks `verify()`.

Rows are never edited. A correction is a new row that references the old one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
LEDGER_RELATIVE_PATH = Path("research_ledger") / "scout_ledger.jsonl"


def _main_checkout_root_path() -> Path:
    """Root of the MAIN checkout, even when running from a git worktree.

    Trials written into a feature worktree's copy would vanish with the worktree,
    so the default ledger always lives in the main checkout.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT_PATH, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return REPO_ROOT_PATH
    return Path(result.stdout.strip()).parent


MAIN_CHECKOUT_ROOT_PATH = _main_checkout_root_path()
DEFAULT_LEDGER_PATH = MAIN_CHECKOUT_ROOT_PATH / LEDGER_RELATIVE_PATH
GENESIS_HASH_STR = "0" * 64

ROW_TYPE_TUPLE = (
    "registration",
    "trial",
    "station_verdict",
    "test_result",
    "vault_opening",
    "gate_result",
    "note",
)
CHAIN_FIELD_TUPLE = ("row_id_int", "utc_ts_str", "prev_row_hash_str", "row_hash_str")


class LedgerIntegrityError(RuntimeError):
    """The chain is broken: a row was edited, removed, inserted or reordered."""


def canonical_json_str(value_obj) -> str:
    """Deterministic JSON: sorted keys, compact, UTF-8, and no NaN or infinity."""
    return json.dumps(value_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def row_hash_str(row_dict: dict) -> str:
    unhashed_dict = {key: value for key, value in row_dict.items() if key != "row_hash_str"}
    return hashlib.sha256(canonical_json_str(unhashed_dict).encode("utf-8")).hexdigest()


def _validate_payload(value_obj, path_str: str = "row") -> None:
    """Reject values that would not survive a JSON round trip unchanged.

    Non-finite floats have no JSON form. Non-string dict keys are hashed in numeric
    order but written as strings, so the reloaded row would sort differently and
    its hash would never match again.
    """
    if isinstance(value_obj, float) and not math.isfinite(value_obj):
        raise ValueError(f"{path_str} is {value_obj}; store missing values as null.")
    if isinstance(value_obj, dict):
        for key, child_obj in value_obj.items():
            if not isinstance(key, str):
                raise ValueError(f"{path_str} has a non-string key {key!r}; use string keys.")
            _validate_payload(child_obj, f"{path_str}.{key}")
    elif isinstance(value_obj, (list, tuple)):
        for idx_int, child_obj in enumerate(value_obj):
            _validate_payload(child_obj, f"{path_str}[{idx_int}]")


class Ledger:
    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER_PATH, lock_timeout_float: float = 30.0):
        self.ledger_path = Path(ledger_path)
        self.lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        self.lock_timeout_float = lock_timeout_float

    # ------------------------------------------------------------------ reading
    def rows(self, row_type_str: str | None = None) -> Iterator[dict]:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as ledger_file:
            for line_number_int, line_str in enumerate(ledger_file, start=1):
                line_str = line_str.strip()
                if not line_str:
                    continue
                try:
                    row_dict = json.loads(line_str)
                except json.JSONDecodeError as error:
                    # A crash in the middle of a write leaves a partial last line.
                    raise LedgerIntegrityError(f"Line {line_number_int} is not valid JSON ({error.msg}).") from None
                if row_type_str is None or row_dict.get("row_type_str") == row_type_str:
                    yield row_dict

    def verify(self) -> int:
        """Walk the whole chain; return the row count or raise LedgerIntegrityError."""
        expected_prev_hash_str = GENESIS_HASH_STR
        expected_row_id_int = 1
        for row_dict in self.rows():
            row_id_obj = row_dict.get("row_id_int")
            if row_id_obj != expected_row_id_int:
                raise LedgerIntegrityError(f"Row id {row_id_obj} found where {expected_row_id_int} was expected.")
            if row_dict.get("prev_row_hash_str") != expected_prev_hash_str:
                raise LedgerIntegrityError(f"Row {row_id_obj}: previous-hash link is broken.")
            if row_hash_str(row_dict) != row_dict.get("row_hash_str"):
                raise LedgerIntegrityError(f"Row {row_id_obj}: content does not match its hash.")
            expected_prev_hash_str = row_dict["row_hash_str"]
            expected_row_id_int += 1
        return expected_row_id_int - 1

    def _last_row(self) -> dict | None:
        last_row_dict = None
        for row_dict in self.rows():
            last_row_dict = row_dict
        return last_row_dict

    # ------------------------------------------------------------------ writing
    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        deadline_float = time.monotonic() + self.lock_timeout_float
        while True:
            try:
                lock_fd_int = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):
                # Windows raises PermissionError while another writer's lock file is being deleted.
                if time.monotonic() > deadline_float:
                    raise TimeoutError(
                        f"Ledger lock {self.lock_path} is held. If no Scout process is running, delete the lock file."
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(lock_fd_int)
            self.lock_path.unlink(missing_ok=True)

    def append(
        self,
        row_type_str: str,
        payload_dict: dict,
        precondition_fn: Callable[["Ledger"], dict | None] | None = None,
    ) -> dict:
        """Append one row and return it with its chain fields filled in.

        `precondition_fn(ledger)` runs inside the lock, after verification, so checks
        such as "this id is not registered yet" cannot race another writer. It may
        raise to refuse the write, or return extra payload fields.
        """
        if row_type_str not in ROW_TYPE_TUPLE:
            raise ValueError(f"Unknown row_type_str {row_type_str!r}; expected one of {ROW_TYPE_TUPLE}.")
        _validate_payload(payload_dict, "payload")

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            # Verify before writing, so a broken chain is never extended.
            self.verify()
            if precondition_fn is not None:
                payload_dict = {**payload_dict, **(precondition_fn(self) or {})}
                _validate_payload(payload_dict, "payload")
            clash_set = set(payload_dict) & set(CHAIN_FIELD_TUPLE + ("row_type_str",))
            if clash_set:
                raise ValueError(f"Payload may not set chain fields: {sorted(clash_set)}.")
            last_row_dict = self._last_row()
            row_dict = {
                "row_id_int": 1 if last_row_dict is None else last_row_dict["row_id_int"] + 1,
                "utc_ts_str": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "prev_row_hash_str": GENESIS_HASH_STR if last_row_dict is None else last_row_dict["row_hash_str"],
                "row_type_str": row_type_str,
                **payload_dict,
            }
            row_dict["row_hash_str"] = row_hash_str(row_dict)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as ledger_file:
                ledger_file.write(canonical_json_str(row_dict) + "\n")
                ledger_file.flush()
                os.fsync(ledger_file.fileno())
        return row_dict
