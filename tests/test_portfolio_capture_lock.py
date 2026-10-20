"""Real OS processes: account capture lock is neither a thread nor a stale flag."""
import multiprocessing
from pathlib import Path

import pytest

from alpha.live.portfolio_capture_lock import portfolio_capture_lock


def _hold_lock(port_int, ready_event, release_event):
    with portfolio_capture_lock("localhost", port_int):
        ready_event.set()
        if not release_event.wait(20):
            raise RuntimeError("Test parent did not release lock")


@pytest.mark.parametrize("terminate_bool", [False, True])
def test_process_lock_covers_loopback_aliases_and_releases_after_exit(terminate_bool):
    # The port is an identity only: no network is opened by this test.
    port_int = 57731 + int(terminate_bool)
    context_obj = multiprocessing.get_context("spawn")
    ready_event = context_obj.Event()
    release_event = context_obj.Event()
    process_obj = context_obj.Process(target=_hold_lock, args=(port_int, ready_event, release_event))
    process_obj.start()
    try:
        assert ready_event.wait(15), "Lock holder failed to start"
        for host_str in ("127.0.0.1", "::1"):
            with pytest.raises(TimeoutError, match="capture busy"):
                with portfolio_capture_lock(host_str, port_int, timeout_seconds_float=0.1):
                    pytest.fail("Other process still owns the same endpoint")
        with portfolio_capture_lock("localhost", port_int + 100, timeout_seconds_float=0.1):
            pass
        if terminate_bool:
            process_obj.terminate()
        else:
            release_event.set()
        process_obj.join(10)
        assert not process_obj.is_alive()
        with portfolio_capture_lock("::1", port_int, timeout_seconds_float=0.1):
            pass
    finally:
        if process_obj.is_alive():
            process_obj.terminate()
            process_obj.join(5)


def test_lock_persists_and_exception_releases_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(ValueError, match="capture failed"):
        with portfolio_capture_lock("localhost", 57733):
            raise ValueError("capture failed")
    lock_path_list = list((tmp_path / ".alpha_super" / "locks").glob("*.lock"))
    assert len(lock_path_list) == 1
    with portfolio_capture_lock("127.0.0.1", 57733, timeout_seconds_float=0.1):
        assert lock_path_list[0].exists()
