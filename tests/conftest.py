"""Default tests use fixtures; real vendor/saved-run checks require an explicit lane."""
from __future__ import annotations

import os
import socket
from tempfile import TemporaryDirectory

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-data-integration", action="store_true", default=False,
                     help="Enable marked local vendor/saved-research integration tests.")


def pytest_configure(config):
    config.addinivalue_line("markers", "data_integration: needs real local vendor data or saved research runs")
    config.addinivalue_line("markers", "integration: needs external research artifacts")
    enabled_bool = config.getoption("--run-data-integration")
    config._alpha_socket_original_dict = {
        method_str: getattr(socket.socket, method_str)
        for method_str in ("listen", "connect", "connect_ex", "sendto")
    }
    config._alpha_socket_original_dict["getaddrinfo"] = socket.getaddrinfo
    if not enabled_bool:
        def deny_session_network(*arg_tuple, **kwarg_dict):
            raise RuntimeError("Offline test session blocked network before test fixtures")

        # Cover imports, collection and session/module fixtures. The per-test
        # fixture below grants access only to that test's own loopback listener.
        patch_obj = pytest.MonkeyPatch()
        config._alpha_network_patch_obj = patch_obj
        for method_str in ("connect", "connect_ex", "sendto"):
            patch_obj.setattr(socket.socket, method_str, deny_session_network)
        patch_obj.setattr(socket, "getaddrinfo", deny_session_network)
    config._alpha_original_env_dict = {
        key_str: os.environ.get(key_str)
        for key_str in ("ALPHA_TEST_DATA_INTEGRATION_BOOL", "NORGATEDATA_ROOT")
    }
    os.environ["ALPHA_TEST_DATA_INTEGRATION_BOOL"] = "true" if enabled_bool else "false"
    if not enabled_bool:
        # Configure before collection: importing the vendor package must not pick
        # the operator's installation. Legacy modules may still mkdir empty local caches.
        config._alpha_norgate_directory_obj = TemporaryDirectory(prefix="alpha-tests-norgate-")
        os.environ["NORGATEDATA_ROOT"] = config._alpha_norgate_directory_obj.name


def pytest_unconfigure(config):
    patch_obj = getattr(config, "_alpha_network_patch_obj", None)
    if patch_obj is not None:
        patch_obj.undo()
    for key_str, value_str in getattr(config, "_alpha_original_env_dict", {}).items():
        if value_str is None:
            os.environ.pop(key_str, None)
        else:
            os.environ[key_str] = value_str
    directory_obj = getattr(config, "_alpha_norgate_directory_obj", None)
    if directory_obj is not None:
        directory_obj.cleanup()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-data-integration"):
        return
    skip_obj = pytest.mark.skip(reason="Real data/artifacts require --run-data-integration; see tests/README.md")
    for item_obj in items:
        if item_obj.get_closest_marker("data_integration") or item_obj.get_closest_marker("integration"):
            item_obj.add_marker(skip_obj)


@pytest.fixture(autouse=True)
def offline_network_guard(request, monkeypatch):
    """Allow only loopback servers opened by this test, never ambient broker/webhooks.

    This guard covers the pytest process, not arbitrary child executables.
    Integration tests opt out only when explicitly selected with the lane flag.
    """
    if request.config.getoption("--run-data-integration") and (
        request.node.get_closest_marker("data_integration") or request.node.get_closest_marker("integration")
    ):
        return
    local_host_set = {"localhost", "127.0.0.1", "::1"}
    listening_port_set = set()
    original_listen_fn = request.config._alpha_socket_original_dict["listen"]
    original_connect_fn = request.config._alpha_socket_original_dict["connect"]
    original_connect_ex_fn = request.config._alpha_socket_original_dict["connect_ex"]
    original_getaddrinfo_fn = request.config._alpha_socket_original_dict["getaddrinfo"]

    def listen_fn(socket_obj, *arg_tuple, **kwarg_dict):
        result_obj = original_listen_fn(socket_obj, *arg_tuple, **kwarg_dict)
        address_obj = socket_obj.getsockname()
        if isinstance(address_obj, tuple) and address_obj[0] in local_host_set:
            listening_port_set.add(address_obj[1])
        return result_obj

    def check_address(address_obj):
        if not (isinstance(address_obj, tuple) and address_obj[0] in local_host_set
                and address_obj[1] in listening_port_set):
            pytest.fail("Offline test attempted a connection outside its own loopback server")

    def connect_fn(socket_obj, address_obj):
        check_address(address_obj)
        return original_connect_fn(socket_obj, address_obj)

    def connect_ex_fn(socket_obj, address_obj):
        check_address(address_obj)
        return original_connect_ex_fn(socket_obj, address_obj)

    def getaddrinfo_fn(host_str, *arg_tuple, **kwarg_dict):
        if host_str not in local_host_set and host_str is not None:
            pytest.fail("Offline test attempted external DNS")
        return original_getaddrinfo_fn(host_str, *arg_tuple, **kwarg_dict)

    def sendto_fn(socket_obj, *arg_tuple, **kwarg_dict):
        pytest.fail("Offline test attempted a datagram connection")

    monkeypatch.setattr(socket.socket, "sendto", sendto_fn)
    monkeypatch.setattr(socket.socket, "listen", listen_fn)
    monkeypatch.setattr(socket.socket, "connect", connect_fn)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex_fn)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo_fn)
