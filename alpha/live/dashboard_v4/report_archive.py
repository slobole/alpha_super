"""Immutable monthly report snapshots; only an explicit issue action saves one.

The caller supplies self-contained HTML without controls or tokens, plus the
issue timestamp and reviewed evidence. A digest verifies these exact saved
bytes; it is not proof of source authenticity or independent audit.
"""

import hashlib
import json
from pathlib import Path
import re


ARCHIVE_VERSION_INT = 1
ARCHIVE_BYTE_LIMIT_INT = 4 * 1024 * 1024
ARCHIVE_ID_RE = re.compile(r"[0-9a-f]{64}")


class ReportArchiveError(ValueError):
    """A report could not be saved or its saved snapshot could not be verified."""


class ReportArchiveNotFoundError(ReportArchiveError):
    """The requested saved report does not exist."""


class ReportArchiveStore:
    """File snapshots beside the event log; isolated in-memory snapshots for demos."""

    def __init__(self, provider_obj, *, memory_bool=False):
        self._provider_obj = provider_obj
        self._memory_bool = memory_bool
        self._memory_dict = {}

    def _directory_obj(self):
        for source_obj in (self._provider_obj, getattr(self._provider_obj, "app_obj", lambda: None)()):
            path_str = getattr(source_obj, "event_log_path_str", None) if source_obj is not None else None
            if isinstance(path_str, str) and path_str:
                return Path(path_str).parent / "monthly_reports"
        raise ReportArchiveError("The monthly report archive location is not configured.")

    def _read_bytes(self, archive_id_str):
        if not isinstance(archive_id_str, str) or not ARCHIVE_ID_RE.fullmatch(archive_id_str):
            raise ReportArchiveError("Invalid saved report identifier.")
        if self._memory_bool:
            if archive_id_str not in self._memory_dict:
                raise ReportArchiveNotFoundError("The saved monthly report was not found.")
            payload_bytes = self._memory_dict[archive_id_str]
        else:
            try:
                with (self._directory_obj() / (archive_id_str + ".json")).open("rb") as source_obj:
                    payload_bytes = source_obj.read(ARCHIVE_BYTE_LIMIT_INT + 1)
            except FileNotFoundError as error_obj:
                raise ReportArchiveNotFoundError("The saved monthly report was not found.") from error_obj
            except OSError as error_obj:
                raise ReportArchiveError("The saved monthly report could not be read.") from error_obj
        if len(payload_bytes) > ARCHIVE_BYTE_LIMIT_INT or hashlib.sha256(payload_bytes).hexdigest() != archive_id_str:
            raise ReportArchiveError("The saved monthly report failed its integrity check.")
        return payload_bytes

    def save(self, html_str, evidence_dict):
        """Freeze HTML and evidence; identical inputs reuse the same immutable ID."""
        if not isinstance(html_str, str) or not html_str.strip() or not isinstance(evidence_dict, dict):
            raise ReportArchiveError("A complete rendered report and its evidence are required.")
        try:
            payload_bytes = json.dumps({"version_int": ARCHIVE_VERSION_INT, "html_str": html_str,
                "evidence_dict": evidence_dict}, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as error_obj:
            raise ReportArchiveError("The monthly report evidence could not be saved.") from error_obj
        if len(payload_bytes) > ARCHIVE_BYTE_LIMIT_INT:
            raise ReportArchiveError("The monthly report is too large to archive.")
        archive_id_str = hashlib.sha256(payload_bytes).hexdigest()
        if self._memory_bool:
            self._memory_dict.setdefault(archive_id_str, payload_bytes)
        else:
            directory_obj = self._directory_obj()
            path_obj = directory_obj / (archive_id_str + ".json")
            created_bool = False
            try:
                directory_obj.mkdir(exist_ok=True)
                with path_obj.open("xb") as output_obj:
                    created_bool = True
                    output_obj.write(payload_bytes)
            except FileExistsError:
                # Never overwrite an issued file, even if its bytes are damaged.
                pass
            except OSError as error_obj:
                if created_bool:
                    try:
                        path_obj.unlink()
                    except OSError:
                        pass
                raise ReportArchiveError("The monthly report could not be archived. No issued report is available from this attempt.") from error_obj
        if self._read_bytes(archive_id_str) != payload_bytes:
            raise ReportArchiveError("An existing saved report does not match this report.")
        return archive_id_str

    def read(self, archive_id_str):
        """Return a fresh verified payload; reads never create or change an archive."""
        try:
            payload_dict = json.loads(self._read_bytes(archive_id_str).decode("utf-8"))
        except (ValueError, RecursionError) as error_obj:
            if isinstance(error_obj, ReportArchiveError):
                raise
            raise ReportArchiveError("The saved monthly report has an invalid format.") from error_obj
        if (not isinstance(payload_dict, dict) or set(payload_dict) != {"version_int", "html_str", "evidence_dict"}
                or type(payload_dict["version_int"]) is not int or payload_dict["version_int"] != ARCHIVE_VERSION_INT
                or not isinstance(payload_dict["html_str"], str) or not payload_dict["html_str"].strip()
                or not isinstance(payload_dict["evidence_dict"], dict)):
            raise ReportArchiveError("The saved monthly report has an unsupported format.")
        return payload_dict
