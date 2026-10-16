"""Run the shared audit checks, excluding already-passing whole test modules."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as element_tree_module

REPO_PATH = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_PATH / "results/review/wired_signoff_20261015"


def main() -> None:
    passed_module_set = set()
    for report_name_str in ("ops_offline_tests.xml", "mr_offline_tests_final.xml", "taa_ndx_focused_tests_verified.xml", "taa_ndx_integrity_tests.xml"):
        report_path = OUTPUT_PATH / report_name_str
        if not report_path.exists():
            continue
        report_obj = element_tree_module.parse(report_path)
        if report_obj.findall(".//failure") or report_obj.findall(".//error"):
            continue
        for case_obj in report_obj.findall(".//testcase"):
            class_name_str = case_obj.attrib.get("classname", "")
            if class_name_str.startswith("tests.test_"):
                passed_module_set.add(class_name_str.split(".")[1] + ".py")
    pattern_tuple = ("test_live*.py", "test_norgate*.py", "test_*norgate_snapshot*.py", "test_dashboard_v4*.py", "test_execution_timing*.py")
    target_set = {test_path for pattern_str in pattern_tuple for test_path in (REPO_PATH / "tests").glob(pattern_str)}
    target_set.update(REPO_PATH / "tests" / name_str for name_str in ("test_strategy_registry.py", "test_bench.py", "test_strategy_dividend_accounting.py", "test_corporate_action_signal_invariance.py", "test_strategy_readiness_audit_review_live_20260928.py", "test_strategy_readiness_audit_live_path_20260928.py"))
    target_list = [str(test_path.relative_to(REPO_PATH)) for test_path in sorted(target_set) if test_path.name not in passed_module_set]
    (OUTPUT_PATH / "remaining_test_targets.json").write_text(__import__("json").dumps(target_list, indent=2), encoding="utf-8")
    print(f"Running {len(target_list)} shared test modules; excluding {len(passed_module_set)} previously passing modules.", flush=True)
    result_obj = subprocess.run([sys.executable, "-m", "pytest", *target_list, "-q", "--basetemp", str(OUTPUT_PATH / "shared_pytest_temp"), "--junitxml", str(OUTPUT_PATH / "shared_tests.xml")], cwd=REPO_PATH)
    raise SystemExit(result_obj.returncode)


if __name__ == "__main__":
    main()
