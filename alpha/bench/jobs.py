"""A small background job runner for Bench.

A backtest takes minutes, so a web request cannot block on it. When you click
a run button Bench spawns the existing CLI as a subprocess, captures its output
to a log file, and tracks its status in memory. This is the only genuinely
"backend" part of Bench, and it is deliberately tiny:

  * jobs run in worker threads gated by a small concurrency semaphore,
  * stdout+stderr stream to ``results/_bench/jobs/{job_id}.log``,
  * a ``{job_id}.json`` sidecar persists status so the Jobs page survives a
    restart (a job still "running" when Bench was last killed is marked
    ``unknown`` on the next start — we cannot reattach to the child to learn its
    real outcome, and on Windows the child may even still be running).

Subprocesses inherit Bench's environment and run with ``cwd`` = repo root, so
they behave exactly like the same command typed in the terminal.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
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
# Distinct from "failed": the run did not produce a verdict, it was stopped. A
# cancelled job's artifacts are whatever the child had already written, which
# for a multi-analysis run may be a complete vanilla and no capacity at all.
STATUS_CANCELLED_STR = "cancelled"
# Used for jobs that were still active when a previous Bench process exited. We
# cannot reattach to the child to learn its real outcome — and on Windows the
# child can outlive the parent — so "unknown" is the honest label, not "done".
STATUS_UNKNOWN_STR = "unknown"

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
    pid_int: int | None = None  # OS pid of the child, persisted for post-restart forensics
    # Recomputed on every listing and stripped before persisting — where a job
    # sits in line is a live fact about this process, not part of its record.
    queue_position_int: int | None = field(default=None, compare=False)

    @property
    def is_active_bool(self) -> bool:
        return self.status_str in _ACTIVE_STATUS_SET

    @property
    def is_cancellable_bool(self) -> bool:
        return self.is_active_bool

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
    # Artifact writers preserve fractional seconds. BENCH job evidence must do
    # the same or a report saved late in the completion second can appear newer
    # than the PASS that produced it.
    return datetime.now().isoformat(timespec="microseconds")


def terminate_process_tree_fn(process_obj: subprocess.Popen) -> None:
    """Kill a launched job and everything it spawned.

    *** CRITICAL*** ``Popen.terminate()`` alone is not enough here. The analysis
    runner is a launcher: ``run_strategy_analysis.py`` spawns
    ``run_strategy.py``, ``run_capacity_analysis.py`` and friends as its own
    children, and those are what actually run the backtest. Killing only the
    parent would return the UI to "cancelled" while the real work kept burning
    CPU with no way left to reach it — a cancel button that lies.

    Windows has no process groups we can signal, so the tree is torn down with
    ``taskkill /T``. POSIX gets its own session at spawn time and takes a group
    SIGTERM.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process_obj.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process_obj.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Already gone, or never ours to signal — either way nothing to stop.
        process_obj.terminate()


class JobManager:
    """Owns the job registry, the worker threads, and the on-disk sidecars."""

    def __init__(self, max_concurrency_int: int = 2) -> None:
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency_int))
        self._job_by_id_dict: dict[str, Job] = {}
        # Live child handles, and the ids the operator asked us to stop. Both
        # are process-local: a cancel cannot reach a job from a previous Bench.
        self._process_by_id_dict: dict[str, subprocess.Popen] = {}
        self._cancelled_id_set: set[str] = set()
        self._worker_thread_list: list[threading.Thread] = []
        JOBS_DIR_PATH.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()

    # ── persistence ──────────────────────────────────────────────────────

    def _sidecar_path(self, job_id_str: str) -> Path:
        return JOBS_DIR_PATH / f"{job_id_str}.json"

    def _persist(self, job_obj: Job) -> None:
        payload_dict = asdict(job_obj)
        payload_dict.pop("queue_position_int", None)
        self._sidecar_path(job_obj.job_id_str).write_text(
            json.dumps(payload_dict, indent=2), encoding="utf-8"
        )

    def _load_persisted_jobs(self) -> None:
        for sidecar_path in JOBS_DIR_PATH.glob("*.json"):
            try:
                payload_dict = json.loads(sidecar_path.read_text(encoding="utf-8"))
                job_obj = Job(**payload_dict)
            except (OSError, ValueError, TypeError):
                continue
            # This is a fresh process: we have no handle on whatever the previous
            # Bench launched. The child may have finished, or may still be running
            # detached (common on Windows). We cannot know, so mark it "unknown"
            # rather than claiming it was interrupted.
            if job_obj.status_str in _ACTIVE_STATUS_SET:
                job_obj.status_str = STATUS_UNKNOWN_STR
                job_obj.ended_at_str = job_obj.ended_at_str or _now_iso_str()
                self._persist(job_obj)
            self._job_by_id_dict[job_obj.job_id_str] = job_obj

    # ── public API ───────────────────────────────────────────────────────

    def submit(self, label_str: str, target_str: str, kind_str: str, command_list: list[str]) -> Job:
        normalized_command_list = list(command_list)
        job_id_str = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        job_obj = Job(
            job_id_str=job_id_str,
            label_str=label_str,
            target_str=target_str,
            kind_str=kind_str,
            command_list=normalized_command_list,
            status_str=STATUS_QUEUED_STR,
            created_at_str=_now_iso_str(),
            log_rel_str=f"results/_bench/jobs/{job_id_str}.log",
        )
        with self._lock:
            # Guard against double-submits (impatient double-click, refresh): if an
            # identical command is already queued/running, return that job instead
            # of launching a duplicate.
            for existing_job_obj in self._job_by_id_dict.values():
                if existing_job_obj.is_active_bool and existing_job_obj.command_list == normalized_command_list:
                    return existing_job_obj
            self._job_by_id_dict[job_id_str] = job_obj
            self._persist(job_obj)

        worker_thread = threading.Thread(target=self._run_job, args=(job_id_str,), daemon=True)
        with self._lock:
            self._worker_thread_list.append(worker_thread)
        worker_thread.start()
        return job_obj

    def wait_for_workers(self, timeout_float: float = 30.0) -> bool:
        """Block until every worker thread has finished. Returns whether it did.

        A worker writes its final sidecar after the job ends, so anything that
        redirects JOBS_DIR_PATH — tests, most of all — must wait here first.
        Otherwise a thread outlives the redirect and persists its record into
        the real results tree, putting test jobs in the operator's console.
        """
        with self._lock:
            worker_thread_list = list(self._worker_thread_list)
        deadline_float = time.monotonic() + timeout_float
        for worker_thread in worker_thread_list:
            remaining_float = deadline_float - time.monotonic()
            if remaining_float <= 0:
                return False
            worker_thread.join(timeout=remaining_float)
            if worker_thread.is_alive():
                return False
        return True

    def list_jobs(self) -> list[Job]:
        with self._lock:
            job_list = list(self._job_by_id_dict.values())
        job_list.sort(key=lambda job_obj: job_obj.created_at_str, reverse=True)
        # Concurrency is small, so a queued job can wait a long time behind two
        # backtests. Say where it is in line rather than showing a bare "queued".
        queued_job_list = sorted(
            (job_obj for job_obj in job_list if job_obj.status_str == STATUS_QUEUED_STR),
            key=lambda job_obj: job_obj.created_at_str,
        )
        for queue_index_int, queued_job_obj in enumerate(queued_job_list, start=1):
            queued_job_obj.queue_position_int = queue_index_int
        return job_list

    def get_job(self, job_id_str: str) -> Job | None:
        with self._lock:
            return self._job_by_id_dict.get(job_id_str)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job_obj in self._job_by_id_dict.values() if job_obj.is_active_bool)

    def cancel(self, job_id_str: str) -> bool:
        """Stop a queued or running job. Returns whether anything was stopped.

        A queued job is marked cancelled and its worker declines to launch. A
        running job has its whole process tree torn down; the worker then sees
        the id in the cancelled set and records "cancelled" rather than reading
        the kill's non-zero exit code as a failed backtest.
        """
        with self._lock:
            job_obj = self._job_by_id_dict.get(job_id_str)
            if job_obj is None or not job_obj.is_active_bool:
                return False
            self._cancelled_id_set.add(job_id_str)
            process_obj = self._process_by_id_dict.get(job_id_str)

        if process_obj is not None:
            terminate_process_tree_fn(process_obj)
            # The worker thread is blocked in wait(); it writes the final status
            # once the child dies, so we do not touch the status here.
            return True

        # Still queued — no child exists yet. Close it out now so the row does
        # not sit at "queued" until a semaphore slot happens to free up.
        self._set_status(
            job_id_str,
            status_str=STATUS_CANCELLED_STR,
            ended_at_str=_now_iso_str(),
        )
        return True

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
            # Cancelled while it sat in the queue: never launch it.
            with self._lock:
                if job_id_str in self._cancelled_id_set:
                    return

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
                    process_obj = subprocess.Popen(
                        job_obj.command_list,
                        cwd=str(REPO_ROOT_PATH),
                        stdout=log_file_obj,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                        # POSIX only: give the child its own session so cancel
                        # can signal the whole tree. Windows tears the tree down
                        # with taskkill instead — see terminate_process_tree_fn.
                        **({} if sys.platform == "win32" else {"start_new_session": True}),
                    )
                    # *** CRITICAL*** Re-check cancellation now that the child
                    # exists. A cancel arriving between the queue gate above and
                    # this line finds no process handle to kill, so it takes the
                    # "still queued" branch and marks the job cancelled — while
                    # this thread goes on to launch the backtest anyway. Without
                    # this second check the UI would show a stopped job whose
                    # analysis is still running and still writing artifacts.
                    with self._lock:
                        self._process_by_id_dict[job_id_str] = process_obj
                        already_cancelled_bool = job_id_str in self._cancelled_id_set
                    if already_cancelled_bool:
                        terminate_process_tree_fn(process_obj)
                    # Record the pid before we block, so a restart mid-run has it.
                    self._set_status(job_id_str, pid_int=process_obj.pid)
                    return_code_int = process_obj.wait()

                with self._lock:
                    self._process_by_id_dict.pop(job_id_str, None)
                    was_cancelled_bool = job_id_str in self._cancelled_id_set
                if was_cancelled_bool:
                    # The non-zero code here is our own kill, not a verdict on
                    # the strategy. Recording it as "failed" would put a red row
                    # against a run nobody ever judged.
                    self._set_status(
                        job_id_str,
                        status_str=STATUS_CANCELLED_STR,
                        return_code_int=return_code_int,
                        ended_at_str=_now_iso_str(),
                    )
                    return
                self._set_status(
                    job_id_str,
                    status_str=STATUS_PASSED_STR if return_code_int == 0 else STATUS_FAILED_STR,
                    return_code_int=return_code_int,
                    ended_at_str=_now_iso_str(),
                )
            except Exception as exception_obj:  # noqa: BLE001 — surface any launch failure to the UI
                with self._lock:
                    self._process_by_id_dict.pop(job_id_str, None)
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
