"""Run canonical wrappers with frozen loader inputs; provider cache misses fail."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))
OUTPUT_PATH = REPO_PATH / "results/review/wired_signoff_20261015"


def run_one(key_str: str) -> tuple[str, int]:
    log_path = OUTPUT_PATH / f"baseline_worker_{key_str}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result_obj = subprocess.run([sys.executable, "scripts/research/wired_signoff_20261015/audit_runner.py", "baselines",
            "--only", key_str, "--check-repeat", "--cached-inputs", str(OUTPUT_PATH / "inputs/native_loader_cache.pkl.gz")],
            cwd=REPO_PATH, stdout=log_file, stderr=subprocess.STDOUT)
    summary_path = OUTPUT_PATH / "baselines" / key_str / "summary.json"
    summary_dict = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    valid_bool = (result_obj.returncode == 0 and summary_dict.get("status") == "pass"
        and summary_dict.get("deterministic_equity") is True and summary_dict.get("capital_honored") is True
        and summary_dict.get("deterministic_economic_transactions") is True
        and summary_dict.get("actual_capital_base") == 1_000_000
        and summary_dict.get("actual_double_capital_base") == 2_000_000)
    print(f"CACHED_BASELINE_END {key_str}: {'PASS' if valid_bool else 'CHECK_LOG'}", flush=True)
    return key_str, 0 if valid_bool else 1


def main() -> None:
    from alpha.strategy_registry import wired_import_tuple, module_import_str
    cache_path = OUTPUT_PATH / "inputs/native_loader_cache.pkl.gz"
    if not cache_path.exists():
        raise FileNotFoundError("Run audit_runner.py prepare before offline workers.")
    manifest_dict = json.loads((OUTPUT_PATH / "frozen_manifest.json").read_text(encoding="utf-8"))
    from audit_runner import file_hash_str
    def verify_source_files() -> None:
        changed_list = [row_dict["path"] for row_dict in manifest_dict["source_files"]
            if file_hash_str(REPO_PATH / row_dict["path"]) != row_dict["sha256"]]
        if changed_list:
            raise RuntimeError(f"Production sources changed since freeze: {changed_list}")
    verify_source_files()
    key_list = [module_import_str(reference_str).rsplit(".", 1)[-1] for reference_str in wired_import_tuple()]
    def priority_int(key_str: str) -> int:
        if "strategy_taa_df" in key_str:
            return 0
        if "strategy_mo_" in key_str:
            return 1
        if "vix_gated" in key_str:
            return 4
        return 3 if "core5" in key_str else 2
    key_list.sort(key=priority_int)
    # Fixed bound: isolated workers parallelize CPU work; actual native reads
    # happened once in prepare. Every missing cache request is a hard error.
    with ThreadPoolExecutor(max_workers=3) as executor_obj:
        future_list = [executor_obj.submit(run_one, key_str) for key_str in key_list]
        outcome_list = [future_obj.result() for future_obj in as_completed(future_list)]
    verify_source_files()
    summary_list = [json.loads((OUTPUT_PATH / "baselines" / key_str / "summary.json").read_text(encoding="utf-8")) for key_str in key_list]
    (OUTPUT_PATH / "baseline_summary.json").write_text(json.dumps(summary_list, indent=2), encoding="utf-8")
    raise SystemExit(any(exit_int for _, exit_int in outcome_list))


if __name__ == "__main__":
    main()
