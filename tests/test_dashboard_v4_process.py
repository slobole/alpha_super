"""Dashboard process settings: math-library thread cap and a quiet poll log."""

import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT_PATH_OBJ = Path(__file__).resolve().parents[1]


def _main_module(monkeypatch):
    # Importing the entry point sets thread defaults; keep this test process unchanged.
    for key_str in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key_str, os.environ.get(key_str, "unchanged"))
    from alpha.live.dashboard_v4 import __main__ as main_module
    return main_module


def test_entry_point_caps_math_threads_before_numpy_loads():
    env_dict = {key_str: value_str for key_str, value_str in os.environ.items()
        if key_str not in {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}}
    env_dict["PYTHONPATH"] = str(ROOT_PATH_OBJ)
    script_str = ("import sys, os\n"
        "import alpha.live.dashboard_v4.__main__\n"
        "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])\n")
    result_obj = subprocess.run([sys.executable, "-c", script_str], cwd=ROOT_PATH_OBJ, env=env_dict,
        capture_output=True, text=True, timeout=120)
    assert result_obj.returncode == 0, result_obj.stderr
    assert result_obj.stdout.split() == ["1", "1"]


def test_entry_point_keeps_an_operator_thread_setting():
    env_dict = {**os.environ, "OPENBLAS_NUM_THREADS": "4", "PYTHONPATH": str(ROOT_PATH_OBJ)}
    script_str = "import os\nimport alpha.live.dashboard_v4.__main__\nprint(os.environ['OPENBLAS_NUM_THREADS'])\n"
    result_obj = subprocess.run([sys.executable, "-c", script_str], cwd=ROOT_PATH_OBJ, env=env_dict,
        capture_output=True, text=True, timeout=120)
    assert result_obj.returncode == 0, result_obj.stderr
    assert result_obj.stdout.strip() == "4"


@pytest.mark.parametrize("request_line_str, code_str, kept_bool", [
    ("GET /overview/refresh?period=All HTTP/1.1", "200", False),
    ("GET /pods/pod_a/refresh?period=All&cycle=&tab= HTTP/1.1", "200", False),
    ("GET /performance/status HTTP/1.1", "200", False),
    ("GET /tools/status HTTP/1.1", "304", False),
    ("GET /console/pod_a/tail?cursor=v1 HTTP/1.1", "200", False),
    ("GET /overview/refresh HTTP/1.1", "500", True),
    ("GET /console/pod_a/tail HTTP/1.1", "429", True),
    ("GET / HTTP/1.1", "200", True),
    ("GET /pods/pod_a HTTP/1.1", "200", True),
    ("GET /console?pod=pod_a HTTP/1.1", "200", True),
    ("POST /overview/refresh HTTP/1.1", "403", True),
])
def test_quiet_filter_drops_only_successful_polls(monkeypatch, request_line_str, code_str, kept_bool):
    main_module = _main_module(monkeypatch)
    record_obj = logging.LogRecord("werkzeug", logging.INFO, __file__, 1, '"%s" %s %s',
        (request_line_str, code_str, "-"), None)
    assert main_module.QuietPollFilter().filter(record_obj) is kept_bool


def test_quiet_filter_keeps_records_without_request_arguments(monkeypatch):
    main_module = _main_module(monkeypatch)
    record_obj = logging.LogRecord("werkzeug", logging.INFO, __file__, 1, "Running on %s", ("http://x",), None)
    assert main_module.QuietPollFilter().filter(record_obj) is True
