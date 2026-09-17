from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_norgate_server_task as task_module


@pytest.fixture
def task_checkout(tmp_path, monkeypatch):
    repo_path_obj = tmp_path / "checkout with spaces"
    (repo_path_obj / "scripts").mkdir(parents=True)
    monkeypatch.setattr(task_module, "__file__", str(repo_path_obj / "scripts" / "run_norgate_server_task.py"))
    config_env_dict = {
        "NORGATE_API_TOKEN": "task-secret-value",
        "NORGATE_SERVICE_ROOT": str(repo_path_obj / "artifact store"),
        "NORGATE_API_HOST": "127.0.0.2",
        "NORGATE_API_PORT": "8799",
    }
    for key_str in config_env_dict:
        monkeypatch.setenv(key_str, "stale-inherited-value")
    (repo_path_obj / "config.env").write_text(
        "\n".join(f"{key_str}={value_str}" for key_str, value_str in config_env_dict.items()), encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return repo_path_obj, config_env_dict


def test_task_loads_checkout_config_before_api_and_preserves_arguments(task_checkout, monkeypatch, capsys):
    repo_path_obj, config_env_dict = task_checkout
    argument_list = ["task", "--host", "127.0.0.3", "--start-date", "1990-01-01"]
    monkeypatch.setattr(sys, "argv", argument_list)
    call_list = []

    def fake_serve_main():
        call_list.append(True)
        assert Path.cwd() == repo_path_obj
        assert {key_str: os.environ[key_str] for key_str in config_env_dict} == config_env_dict
        assert sys.argv == argument_list
        return 17

    monkeypatch.setitem(sys.modules, "scripts.export_norgate_snapshot", SimpleNamespace(
        SUPPORTED_EOD_PROFILE_TUPLE=("norgate_eod_core5",),
    ))
    monkeypatch.setitem(sys.modules, "scripts.serve_norgate_snapshot_api", SimpleNamespace(main=fake_serve_main))
    assert task_module.main() == 17
    assert call_list == [True]
    captured_obj = capsys.readouterr()
    assert "norgate_eod_core5" in captured_obj.out
    assert str(repo_path_obj) in captured_obj.out
    assert config_env_dict["NORGATE_API_TOKEN"] not in captured_obj.out + captured_obj.err


@pytest.mark.parametrize("invalid_config_str", [None, "invalid line", "NORGATE_API_TOKEN=\n", "NORGATE_API_TOKEN=present\n"])
def test_task_rejects_missing_or_incomplete_config_before_api(task_checkout, monkeypatch, invalid_config_str):
    repo_path_obj, _ = task_checkout
    config_path_obj = repo_path_obj / "config.env"
    if invalid_config_str is None:
        config_path_obj.unlink()
    else:
        config_path_obj.write_text(invalid_config_str, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "scripts.serve_norgate_snapshot_api", None)
    monkeypatch.setitem(sys.modules, "scripts.export_norgate_snapshot", None)
    with pytest.raises((FileNotFoundError, ValueError)):
        task_module.main()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows task wrapper")
@pytest.mark.parametrize("missing_file_str", [None, "python", "bootstrap"])
def test_powershell_wrapper_scopes_process_and_propagates_exit(tmp_path, missing_file_str):
    repo_path_obj = tmp_path / "task checkout with spaces"
    script_dir_path_obj = repo_path_obj / "scripts"
    script_dir_path_obj.mkdir(parents=True)
    python_path_obj = repo_path_obj / ".venv" / "Scripts" / "python.exe"
    python_path_obj.parent.mkdir(parents=True)
    if missing_file_str != "python":
        python_path_obj.touch()
    if missing_file_str != "bootstrap":
        (script_dir_path_obj / "run_norgate_server_task.py").touch()
    source_path_obj = Path(__file__).resolve().parents[1] / "scripts" / "run_norgate_server_task.ps1"
    shutil.copyfile(source_path_obj, script_dir_path_obj / source_path_obj.name)
    harness_path_obj = script_dir_path_obj / "test_harness.ps1"
    harness_path_obj.write_text('''
$ErrorActionPreference = "Stop"
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
          $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru, [switch]$Wait)
    Write-Host ($PSBoundParameters | ConvertTo-Json -Compress)
    return [pscustomobject]@{ ExitCode = 37 }
}
& (Join-Path $PSScriptRoot "run_norgate_server_task.ps1")
exit $LASTEXITCODE
''', encoding="utf-8")
    result_obj = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness_path_obj)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    if missing_file_str:
        assert result_obj.returncode != 0
        assert "Required Norgate task file is missing" in result_obj.stderr
        assert "ArgumentList" not in result_obj.stdout
        return
    assert result_obj.returncode == 37, result_obj.stderr
    invocation_dict = json.loads(result_obj.stdout.strip())
    assert invocation_dict["FilePath"] == str(python_path_obj)
    assert invocation_dict["WorkingDirectory"] == str(repo_path_obj)
    assert invocation_dict["ArgumentList"] == ["-B", "-u", "-m", "scripts.run_norgate_server_task"]
    assert invocation_dict["WindowStyle"] == "Hidden"
    assert invocation_dict["Wait"]["IsPresent"]
    assert invocation_dict["PassThru"]["IsPresent"]
    stdout_path_obj = Path(invocation_dict["RedirectStandardOutput"])
    stderr_path_obj = Path(invocation_dict["RedirectStandardError"])
    assert stdout_path_obj != stderr_path_obj
    assert stdout_path_obj.parent == stderr_path_obj.parent == repo_path_obj / "results" / "logs" / "norgate_api"
