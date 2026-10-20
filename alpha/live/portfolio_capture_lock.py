"""Serialize optional portfolio downloads across local pod processes."""
from contextlib import contextmanager
import errno
from hashlib import sha256
import os
from pathlib import Path
from time import monotonic, sleep


@contextmanager
def portfolio_capture_lock(host_str: str, port_int: int, *, timeout_seconds_float: float = 10.0):
    # Share across accounts, client IDs and release directories for this OS user.
    host_str = host_str.strip().lower()
    if host_str in {"localhost", "127.0.0.1", "::1"}:
        host_str = "loopback"
    endpoint_str = f"{host_str}:{int(port_int)}"
    digest_str = sha256(endpoint_str.encode("utf-8")).hexdigest()[:24]
    lock_root_path_obj = Path.home() / ".alpha_super" / "locks"
    lock_root_path_obj.mkdir(parents=True, exist_ok=True)
    lock_path_obj = lock_root_path_obj / f"portfolio-{digest_str}.lock"
    # Never unlink: waiters must continue to lock the same file after a release.
    with lock_path_obj.open("a+b") as lock_file_obj:
        if os.name == "nt":
            import msvcrt

            def acquire_lock():
                lock_file_obj.seek(0)
                msvcrt.locking(lock_file_obj.fileno(), msvcrt.LK_NBLCK, 1)

            def release_lock():
                lock_file_obj.seek(0)
                msvcrt.locking(lock_file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire_lock():
                fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release_lock():
                fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_UN)

        deadline_float = monotonic() + timeout_seconds_float
        while True:
            try:
                acquire_lock()
                break
            except OSError as exception_obj:
                if exception_obj.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if monotonic() >= deadline_float:
                    raise TimeoutError("IBKR portfolio capture busy") from exception_obj
                sleep(0.05)
        try:
            yield
        finally:
            release_lock()
