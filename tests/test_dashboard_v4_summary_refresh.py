"""The V4 summary never makes a poll wait behind a slow rebuild."""

import threading

import pytest

from alpha.live.dashboard_v3 import data as v3_data_module
from alpha.live.dashboard_v4 import data as data_module
from alpha.live.dashboard_v4.data import LiveDataProvider


class _Clock:
    def __init__(self):
        self.now_float = 1000.0

    def __call__(self):
        return self.now_float


class _Builder:
    """Counts builds; optionally blocks until released or raises."""

    def __init__(self):
        self.call_int = 0
        self.release_obj = threading.Event()
        self.release_obj.set()
        self.started_obj = threading.Event()
        self.error_bool = False
        self.lock_obj = threading.Lock()

    def __call__(self, app_obj):
        with self.lock_obj:
            self.call_int += 1
            call_int = self.call_int
        self.started_obj.set()
        assert self.release_obj.wait(5)
        if self.error_bool:
            raise OSError("event log unreadable")
        return {"as_of_timestamp_str": f"summary-{call_int}", "pod_row_dict_list": []}


@pytest.fixture()
def provider_tuple(monkeypatch):
    clock_obj, builder_obj = _Clock(), _Builder()
    monkeypatch.setattr(data_module, "_monotonic_float", clock_obj)
    monkeypatch.setattr(v3_data_module, "build_dashboard_summary_dict", builder_obj)
    monkeypatch.setattr(data_module, "SUMMARY_WAIT_SECONDS_FLOAT", 0.05)
    provider_obj = LiveDataProvider()
    monkeypatch.setattr(provider_obj, "app_obj", lambda: object())
    return provider_obj, clock_obj, builder_obj


def _wait_idle(provider_obj):
    state_obj = provider_obj._summary_state_obj()
    for _ in range(200):
        with state_obj.lock_obj:
            if state_obj.done_event_obj is None:
                return
        threading.Event().wait(0.01)
    raise AssertionError("rebuild did not finish")


def test_first_request_builds_once_and_cache_is_reused_inside_ttl(provider_tuple):
    provider_obj, clock_obj, builder_obj = provider_tuple
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-1"
    clock_obj.now_float += data_module.SUMMARY_CACHE_SECONDS_FLOAT - 0.1
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-1"
    assert builder_obj.call_int == 1


def test_fast_rebuild_after_ttl_returns_the_new_summary(provider_tuple):
    provider_obj, clock_obj, builder_obj = provider_tuple
    provider_obj.get_summary_dict()
    clock_obj.now_float += data_module.SUMMARY_CACHE_SECONDS_FLOAT
    data_module.SUMMARY_WAIT_SECONDS_FLOAT = 2.0
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-2"


def test_slow_rebuild_serves_last_summary_and_runs_single_flight(provider_tuple):
    provider_obj, clock_obj, builder_obj = provider_tuple
    provider_obj.get_summary_dict()
    builder_obj.release_obj.clear()
    builder_obj.started_obj.clear()
    clock_obj.now_float += 30.0
    result_list = []
    thread_list = [threading.Thread(target=lambda: result_list.append(provider_obj.get_summary_dict()))
        for _ in range(5)]
    for thread_obj in thread_list:
        thread_obj.start()
    for thread_obj in thread_list:
        thread_obj.join(5)
    assert builder_obj.started_obj.wait(5)
    assert [item_dict["as_of_timestamp_str"] for item_dict in result_list] == ["summary-1"] * 5
    assert builder_obj.call_int == 2
    builder_obj.release_obj.set()
    _wait_idle(provider_obj)
    clock_obj.now_float += 1.0
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-2"


def test_failed_rebuild_keeps_last_summary_and_rests_before_retry(provider_tuple):
    provider_obj, clock_obj, builder_obj = provider_tuple
    provider_obj.get_summary_dict()
    builder_obj.error_bool = True
    clock_obj.now_float += 30.0
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-1"
    _wait_idle(provider_obj)
    assert builder_obj.call_int == 2
    # The failed build took no clock time, so the retry is allowed on the next poll.
    builder_obj.error_bool = False
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] in {"summary-1", "summary-3"}
    _wait_idle(provider_obj)
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-3"


def test_rest_window_scales_with_the_last_build_duration(provider_tuple, monkeypatch):
    provider_obj, clock_obj, builder_obj = provider_tuple
    provider_obj.get_summary_dict()

    def slow_build(app_obj):
        builder_obj.call_int += 1
        clock_obj.now_float += 6.0  # the build itself takes six seconds
        return {"as_of_timestamp_str": f"summary-{builder_obj.call_int}", "pod_row_dict_list": []}

    monkeypatch.setattr(v3_data_module, "build_dashboard_summary_dict", slow_build)
    clock_obj.now_float += 30.0
    provider_obj.get_summary_dict()
    _wait_idle(provider_obj)
    assert builder_obj.call_int == 2
    # Data from the start of the build; next build no sooner than 2 x 6 s after its end.
    clock_obj.now_float += 11.0
    provider_obj.get_summary_dict()
    _wait_idle(provider_obj)
    assert builder_obj.call_int == 2
    clock_obj.now_float += 1.0
    provider_obj.get_summary_dict()
    _wait_idle(provider_obj)
    assert builder_obj.call_int == 3


def test_first_build_error_reaches_the_caller_and_is_retried(provider_tuple):
    provider_obj, clock_obj, builder_obj = provider_tuple
    builder_obj.error_bool = True
    with pytest.raises(OSError):
        provider_obj.get_summary_dict()
    builder_obj.error_bool = False
    assert provider_obj.get_summary_dict()["as_of_timestamp_str"] == "summary-2"
