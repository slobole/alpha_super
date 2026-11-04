"""Exercise the default test boundary without contacting external services."""
import socket

import pytest


@pytest.mark.parametrize("method_str", ["connect", "connect_ex"])
def test_ambient_loopback_services_are_blocked(method_str):
    with socket.socket() as socket_obj:
        with pytest.raises(pytest.fail.Exception, match="outside its own loopback"):
            getattr(socket_obj, method_str)(("127.0.0.1", 7497))


def test_external_dns_is_blocked():
    with pytest.raises(pytest.fail.Exception, match="external DNS"):
        socket.getaddrinfo("offline-test.invalid", 443)


def test_a_test_owned_loopback_server_is_allowed():
    with socket.socket() as server_obj, socket.socket() as client_obj:
        server_obj.bind(("127.0.0.1", 0))
        server_obj.listen(1)
        client_obj.connect(server_obj.getsockname())
        accepted_obj, _ = server_obj.accept()
        with accepted_obj:
            client_obj.sendall(b"fixture")
            assert accepted_obj.recv(7) == b"fixture"


def test_datagrams_cannot_bypass_the_guard():
    with socket.socket(type=socket.SOCK_DGRAM) as socket_obj:
        with pytest.raises(pytest.fail.Exception, match="datagram"):
            socket_obj.sendto(b"fixture", ("127.0.0.1", 7497))


def test_data_lane_requires_explicit_opt_in(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    source_path = Path(__file__).with_name("conftest.py")
    (tmp_path / "conftest.py").write_bytes(source_path.read_bytes())
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_lane.py").write_text(
        'import os\nfrom pathlib import Path\nimport pytest\n'
        'def test_fixture():\n    assert True\n'
        '@pytest.mark.data_integration\n'
        'def test_vendor_boundary():\n'
        '    assert os.environ["ALPHA_TEST_DATA_INTEGRATION_BOOL"] == "true"\n'
        '    Path("integration-ran.txt").write_text("fixture only")\n', encoding="utf-8")
    command_list = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--confcutdir", str(tmp_path)]
    environment_dict = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    first_obj = subprocess.run(command_list, cwd=tmp_path, env=environment_dict,
                               capture_output=True, text=True, timeout=30)
    assert first_obj.returncode == 0, first_obj.stdout + first_obj.stderr
    assert "1 passed, 1 skipped" in first_obj.stdout
    assert not (tmp_path / "integration-ran.txt").exists()
    second_obj = subprocess.run(command_list + ["--run-data-integration"], cwd=tmp_path, env=environment_dict,
                                capture_output=True, text=True, timeout=30)
    assert second_obj.returncode == 0, second_obj.stdout + second_obj.stderr
    assert "2 passed" in second_obj.stdout
    assert (tmp_path / "integration-ran.txt").exists()


def test_collection_also_blocks_network_before_fixtures(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    (tmp_path / "conftest.py").write_bytes(Path(__file__).with_name("conftest.py").read_bytes())
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_import.py").write_text(
        'import socket\nimport pytest\n'
        'try:\n    socket.getaddrinfo("collection-test.invalid", 443)\n'
        'except RuntimeError as error_obj:\n    assert "Offline test session" in str(error_obj)\n'
        'else:\n    raise AssertionError("collection guard missing")\n'
        'with socket.socket() as socket_obj:\n'
        '    try:\n        socket_obj.connect(("127.0.0.1", 7497))\n'
        '    except RuntimeError as error_obj:\n        assert "Offline test session" in str(error_obj)\n'
        '    else:\n        raise AssertionError("collection connection guard missing")\n'
        '@pytest.fixture(scope="session", autouse=True)\n'
        'def earlier_fixture():\n'
        '    with pytest.raises(RuntimeError, match="Offline test session"):\n'
        '        socket.getaddrinfo("session-test.invalid", 443)\n'
        'def test_collected():\n    assert True\n', encoding="utf-8")
    result_obj = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--confcutdir", str(tmp_path)],
        cwd=tmp_path, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=30,
    )
    assert result_obj.returncode == 0, result_obj.stdout + result_obj.stderr
    assert "1 passed" in result_obj.stdout
