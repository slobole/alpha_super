"""Proof runners must propagate child failures; no vendor or broker is used."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_PATH = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_PATH / "scripts/research/momentum_capsule_wiring_20261103/run_final_proofs.sh"
GATE_PATH = REPO_PATH / "scripts/research/mr_capsule_liq_build_20261101/gate_ndx_taa.py"


def _bash_path():
    git_bash_path = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash_path.exists():
        return str(git_bash_path)
    bash_str = shutil.which("bash")
    if not bash_str:
        pytest.skip("Bash is required for the shell-wrapper contract")
    return bash_str


@pytest.mark.parametrize("failure_str", ["", "parity", "export", "decisions", "replay", "vanilla", "gate", "compare"])
def test_wrapper_stops_on_first_failed_proof(tmp_path, failure_str):
    wrapper_path = tmp_path / "run.sh"
    wrapper_path.write_text(WRAPPER_PATH.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (tmp_path / "results/research/momentum_capsule_wiring_20261103").mkdir(parents=True)
    fake_python_path = tmp_path / "fake-python"
    fake_python_path.write_text('''#!/usr/bin/env bash
case "$1" in
  */parity_rows.py) STAGE=parity ;;
  */export_norgate_snapshot.py) STAGE=export ;;
  */snapshot_decision.py) STAGE=decisions ;;
  */replay_cut.py) STAGE=replay ;;
  */run_strategy_analysis.py) STAGE=vanilla ;;
  */gate_ndx_taa.py) STAGE=gate; [ "${2:-}" = compare ] && STAGE=compare ;;
  *) exit 91 ;;
esac
echo "$STAGE" >> calls.txt
if [ "$STAGE" = "$FAIL_STAGE" ]; then
  echo "diagnostic-$STAGE" >&2
  exit 23
fi
# Successful tools need not print a particular grep pattern.
echo "completed-$STAGE"
''', encoding="utf-8", newline="\n")
    fake_python_path.chmod(0o755)
    result_obj = subprocess.run(
        [_bash_path(), str(wrapper_path), fake_python_path.as_posix(), "snapshot", "base.json"],
        cwd=tmp_path, env={**os.environ, "FAIL_STAGE": failure_str},
        capture_output=True, text=True, timeout=30,
    )
    output_str = result_obj.stdout + result_obj.stderr
    call_list = (tmp_path / "calls.txt").read_text().splitlines()
    if failure_str:
        assert result_obj.returncode != 0
        assert "PROOFS DONE" not in output_str
        assert f"diagnostic-{failure_str}" in output_str
        assert call_list[-1] == failure_str
    else:
        assert result_obj.returncode == 0, output_str
        assert "PROOFS DONE" in output_str
        assert call_list[-1] == "compare"


@pytest.mark.parametrize("left_dict,right_dict,expected_int", [
    ({"a": {"weight": 0.25}}, {"a": {"weight": 0.25}}, 0),
    ({"a": {"weight": 0.25}}, {"a": {"weight": 0.26}}, 1),
    ({"a": {"weight": 0.25}}, {}, 1),
    ({}, {}, 1),
    ({"a": {"error_str": "refused"}}, {"a": {"error_str": "refused"}}, 0),
])
def test_exact_gate_compare_exit_status(tmp_path, left_dict, right_dict, expected_int):
    path_list = [tmp_path / "base.json", tmp_path / "candidate.json"]
    for path_obj, row_dict in zip(path_list, [left_dict, right_dict]):
        path_obj.write_text(json.dumps({"alpha_file_str": "fixture", "rows": row_dict}), encoding="utf-8")
    result_obj = subprocess.run([sys.executable, "-B", str(GATE_PATH), "compare", *map(str, path_list)],
                                capture_output=True, text=True, timeout=30)
    assert result_obj.returncode == expected_int, result_obj.stdout + result_obj.stderr
    if "error_str" in left_dict.get("a", {}):
        assert "errors on the base 1" in result_obj.stdout


@pytest.fixture
def snapshot_proof_module():
    import importlib.util
    source_path = WRAPPER_PATH.with_name("snapshot_decision.py")
    spec_obj = importlib.util.spec_from_file_location("snapshot_proof_fixture", source_path)
    module_obj = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module_obj)
    return module_obj


@pytest.mark.parametrize("returncode_int,stdout_str,error_bool", [
    (0, 'RESULT {"weight_dict": {}, "path_dict": {}}', False),
    (23, 'RESULT {"weight_dict": {}, "path_dict": {}}', True),
    (0, 'log without a result', True),
    (0, 'RESULT broken json', True),
    (0, 'RESULT {}', True),
    (0, 'RESULT []', True),
    (0, 'RESULT {}\nRESULT {}', True),
])
def test_snapshot_child_exit_and_result_are_both_required(snapshot_proof_module, monkeypatch,
                                                         returncode_int, stdout_str, error_bool):
    from types import SimpleNamespace
    monkeypatch.setattr(snapshot_proof_module.subprocess, "run", lambda *arg_tuple, **kwarg_dict:
                        SimpleNamespace(returncode=returncode_int, stdout=stdout_str, stderr="child diagnostic"))
    result_dict = snapshot_proof_module.run_one("live", "1", "2026-10-30", "unused")
    assert ("error_str" in result_dict) is error_bool
    if returncode_int:
        assert "child diagnostic" in result_dict["error_str"]
        assert result_dict["stdout_tail_str"] == stdout_str


@pytest.mark.parametrize("failed_mode_str", ["live", "direct"])
def test_snapshot_parent_preserves_failure_artifact_and_exits(snapshot_proof_module, monkeypatch,
                                                             tmp_path, capsys, failed_mode_str):
    monkeypatch.setattr(snapshot_proof_module, "OUT_PATH", tmp_path)
    monkeypatch.setattr(snapshot_proof_module, "run_one", lambda mode_str, *arg_tuple:
                        {"error_str": "child failure"} if mode_str == failed_mode_str
                        else {"weight_dict": {}, "path_dict": {}})
    with pytest.raises(SystemExit) as exit_obj:
        snapshot_proof_module.main("unused", "2026-10-30", ["1"])
    assert exit_obj.value.code == 1
    artifact_dict = json.loads((tmp_path / "snapshot_decision_2026-10-30.json").read_text(encoding="utf-8"))
    assert artifact_dict["1"][failed_mode_str]["error_str"] == "child failure"
    captured_obj = capsys.readouterr()
    assert "SNAPSHOT DECISIONS DONE" not in captured_obj.out
    assert "SNAPSHOT DECISIONS FAILED" in captured_obj.err
