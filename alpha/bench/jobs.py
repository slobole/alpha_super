"""A small background job runner for Bench.

A backtest takes minutes, so a web request cannot block on it. When you click
a run button Bench spawns the existing CLI as a subprocess, captures its output
to a log file, and tracks its status in memory. This is the only genuinely
"backend" part of Bench, and it is deliberately tiny:

  * jobs run in worker threads gated by a small concurrency semaphore,
  * stdout+stderr stream to ``results/_bench/jobs/{job_id}.log``,
  * a ``{job_id}.json`` sidecar persists status so the Jobs page survives a
    restart (a job still "running" when Bench was last killed is marked
    ``interrupted`` on the next start, since the OS process is gone).

Subprocesses inherit Bench's environment and run with ``cwd`` = repo root, so
they behave exactly like the same command typed in the terminal.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
JOBS_DIR_PATH = REPO_ROOT_PATH / "results" / "_bench" / "jobs"

STATUS_QUEUED_STR = "queued"
STATUS_RUNNING_STR = "running"
STATUS_PASSED_STR = "passed"
STATUS_FAILED_STR = "failed"
STATUS_ERROR_STR = "error"
STATUS_INTERRUPTED_STR = "interrupted"

_ACTIVE_STATUS_SET = {STATUS_QUEUED_STR, STATUS_RUNNING_STR}


@dataclass
class Job:
    job_id_str: str
    label_str: str
    target_str: str  # strategy stem or portfolio name
    kind_str: str  # "analysis" | "portfolio"
    command_list: list[str]
    status_str: str
    created_at_str: str
    started_at_str: str | None = None
    ended_at_str: str | None = None
    return_code_int: int | None = None
    log_rel_str: str = ""

    @property
    def is_active_bool(self) -> bool:
        return self.status_str in _ACTIVE_STATUS_SET

    @property
    def elapsed_str(self) -> str:
        start_str = self.started_at_str
        if not start_str:
            return "—"
        end_str = self.ended_at_str or datetime.now().isoformat(timespec="seconds")
        try:
            delta_seconds_float = (
                datetime.fromisoformat(end_str) - datetime.fromisoformat(start_str)
            ).total_seconds()
        except ValueError:
            return "—"
        if delta_seconds_float < 60:
            return f"{delta_seconds_float:.0f}s"
        minutes_int, seconds_int = divmod(int(delta_seconds_float), 60)
        return f"{minutes_int}m {seconds_int:02d}s"

    @property
    def command_display_str(self) -> str:
        """Compact command for the UI: ``python`` instead of the full interpreter
        path, and repo-relative script/config paths instead of absolutes."""
        repo_prefix_str = f"{REPO_ROOT_PATH}\\"
        repo_prefix_posix_str = f"{REPO_ROOT_PATH.as_posix()}/"
        pretty_part_list: list[str] = []
        for index_int, part_str in enumerate(self.command_list):
            if index_int == 0 and part_str.lower().endswith(("python.exe", "python")):
                pretty_part_list.append("python")
                continue
            cleaned_str = part_str.replace(repo_prefix_str, "").replace(repo_prefix_posix_str, "")
            pretty_part_list.append(cleaned_str.replace("\\", "/"))
        return subprocess.list2cmdline(pretty_part_list)


def _now_iso_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobManager:
    """Owns the job registry, the worker threads, and the on-disk sidecars."""

    def __init__(self, max_concurrency_int: int = 2) -> None:
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency_int))
        self._job_by_id_dict: dict[str, Job] = {}
        JOBS_DIR_PATH.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()

    # ── persistence ──────────────────────────────────────────────────────

    def _sidecar_path(self, job_id_str: str) -> Path:
        return JOBS_DIR_PATH / f"{job_id_str}.json"

    def _persist(self, job_obj: Job) -> None:
        self._sidecar_path(job_obj.job_id_str).write_text(
            json.dumps(asdict(job_obj), indent=2), encoding="utf-8"
        )

    def _load_persisted_jobs(self) -> None:
        for sidecar_path in JOBS_DIR_PATH.glob("*.json"):
            try:
                payload_dict = json.loads(sidecar_path.read_text(encoding="utf-8"))
                job_obj = Job(**payload_dict)
            except (OSError, ValueError, TypeError):
                continue
            # The owning process is gone, so anything still "active" is stale.
            if job_obj.status_str in _ACTIVE_STATUS_SET:
                job_obj.status_str = STATUS_INTERRUPTED_STR
                job_obj.ended_at_str = job_obj.ended_at_str or _now_iso_str()
                self._persist(job_obj)
            self._job_by_id_dict[job_obj.job_id_str] = job_obj

    # ── public API ───────────────────────────────────────────────────────

    def submit(self, label_str: str, target_str: str, kind_str: str, command_list: list[str]) -> Job:
        job_id_str = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        job_obj = Job(
            job_id_str=job_id_str,
            label_str=label_str,
            target_str=target_str,
            kind_str=kind_str,
            command_list=list(command_list),
            status_str=STATUS_QUEUED_STR,
            created_at_str=_now_iso_str(),
            log_rel_str=f"results/_bench/jobs/{job_id_str}.log",
        )
        with self._lock:
            self._job_by_id_dict[job_id_str] = job_obj
            self._persist(job_obj)

        worker_thread = threading.Thread(target=self._run_job, args=(job_id_str,), daemon=True)
        worker_thread.start()
        return job_obj

    def list_jobs(self) -> list[Job]:
        with self._lock:
            job_list = list(self._job_by_id_dict.values())
        job_list.sort(key=lambda job_obj: job_obj.created_at_str, reverse=True)
        return job_list

    def get_job(self, job_id_str: str) -> Job | None:
        with self._lock:
            return self._job_by_id_dict.get(job_id_str)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job_obj in self._job_by_id_dict.values() if job_obj.is_active_bool)

    def read_log_text(self, job_id_str: str, max_bytes_int: int = 200_000) -> str:
        log_path = JOBS_DIR_PATH / f"{job_id_str}.log"
        if not log_path.is_file():
            return ""
        raw_bytes = log_path.read_bytes()
        if len(raw_bytes) > max_bytes_int:
            raw_bytes = raw_bytes[-max_bytes_int:]
        return raw_bytes.decode("utf-8", errors="replace")

    # ── worker ───────────────────────────────────────────────────────────

    def _set_status(self, job_id_str: str, **field_value_dict) -> None:
        with self._lock:
            job_obj = self._job_by_id_dict.get(job_id_str)
            if job_obj is None:
                return
            for field_name_str, value_obj in field_value_dict.items():
                setattr(job_obj, field_name_str, value_obj)
            self._persist(job_obj)

    def _run_job(self, job_id_str: str) -> None:
        job_obj = self.get_job(job_id_str)
        if job_obj is None:
            return

        with self._semaphore:
            self._set_status(job_id_str, status_str=STATUS_RUNNING_STR, started_at_str=_now_iso_str())
            log_path = JOBS_DIR_PATH / f"{job_id_str}.log"
            try:
                with log_path.open("wb") as log_file_obj:
                    header_str = (
                        f"$ {subprocess.list2cmdline(job_obj.command_list)}\n"
                        f"# started {_now_iso_str()}\n\n"
                    )
                    log_file_obj.write(header_str.encode("utf-8"))
                    log_file_obj.flush()
                    completed_process_obj = subprocess.run(
                        job_obj.command_list,
                        cwd=str(REPO_ROOT_PATH),
                        stdout=log_file_obj,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                        check=False,
                    )
                return_code_int = completed_process_obj.returncode
                self._set_status(
                    job_id_str,
                    status_str=STATUS_PASSED_STR if return_code_int == 0 else STATUS_FAILED_STR,
                    return_code_int=return_code_int,
                    ended_at_str=_now_iso_str(),
                )
            except Exception as exception_obj:  # noqa: BLE001 — surface any launch failure to the UI
                try:
                    with log_path.open("ab") as log_file_obj:
                        log_file_obj.write(f"\n[bench] job failed to launch: {exception_obj}\n".encode("utf-8"))
                except OSError:
                    pass
                self._set_status(
                    job_id_str,
                    status_str=STATUS_ERROR_STR,
                    return_code_int=None,
                    ended_at_str=_now_iso_str(),
                )
