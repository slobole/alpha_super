"""Seal a completed local audit; this cannot activate or alter trading."""
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from build_signoff_report import REPO_PATH, ROOT_PATH, hash_file, verify_production_sources


def main() -> None:
    frozen_dict = json.loads((ROOT_PATH / "frozen_manifest.json").read_text(encoding="utf-8"))
    verify_production_sources(frozen_dict)
    report_dict = json.loads((ROOT_PATH / "audit_report.json").read_text(encoding="utf-8"))
    if not report_dict["complete"] or len(report_dict["cards"]) != 17 or report_dict["tests"]["failed"]:
        raise ValueError("Only a complete17-identity audit with clean selected tests can be sealed.")
    canvas_dict = json.loads((ROOT_PATH / "canvas_typecheck.json").read_text(encoding="utf-8"))
    if canvas_dict["status"] != "pass" or canvas_dict["canvas_sha256"] != hash_file(ROOT_PATH / "wired-signoff.canvas.tsx"):
        raise ValueError("Canvas source must pass its current SDK check.")
    head_str = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_PATH, text=True).strip()
    if head_str != frozen_dict["head"]:
        raise ValueError("Checkout HEAD changed after the audit freeze.")
    subprocess.run([sys.executable, "-m", "compileall", "-q", str(Path(__file__).parent)], check=True)
    subprocess.run(["git", "-c", "core.fsmonitor=false", "diff", "--check", "--", "scripts/research/wired_signoff_20261015"], cwd=REPO_PATH, check=True)
    source_file_list = [source_path for source_path in Path(__file__).parent.iterdir() if source_path.is_file()]
    whitespace_list = [f"{source_path.name}:{line_int}" for source_path in source_file_list
        for line_int, line_str in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1) if line_str.rstrip() != line_str]
    if whitespace_list:
        raise ValueError(f"Audit-owned source trailing whitespace: {whitespace_list}")
    replay_count_dict = {family_str: 0 for family_str in report_dict["family_labels"]}
    for card_dict in report_dict["cards"]:
        replay_dict = card_dict["replay"]
        count_int = replay_dict.get("exact_int", replay_dict.get("native_host_session_count_int", replay_dict.get("passed_session_count_int", 0)))
        replay_count_dict[card_dict["family"]] += count_int
    verification_dict = {"completed_utc": datetime.now(UTC).isoformat(), "tier": "Tier1 research/backtest tooling",
        "production_sources_unchanged": True, "head": head_str, "strategy_count": len(report_dict["cards"]),
        "model_capital_per_strategy": 1_000_000, "repeat_capital": 1_000_000, "capital_scale_check": 2_000_000,
        "selected_unique_tests": {key_str: report_dict["tests"][key_str] for key_str in ("passed", "failed", "skipped")},
        "actual_host_replay_count_by_family": replay_count_dict,
        "reviewers": ["audit_scope_inventory: independent quantitative/statistics/report provenance review",
            "doctor_audit_plan: independent calendar/capacity/cache review", "audit_evidence_plan/mr_review_peer: independent MR guards/capacity/replay review"],
        "audit_tool_findings_fixed": ["HPI exact-input cache and imported engine-binding capture",
            "first-day/subperiod/benchmark provenance and statistics source pins",
            "informative replay/state/date and executed-source publication gates",
            "CORE5 independent NAV/quote shocks and distinction of quantities from monthly weights"],
        "production_findings_fixed": [], "production_recommendations_only": True,
        "checks": ["canonical repeat equity and economic ledger", "actual1M/2M capital bases",
            "production SHA256 freeze", "265 monthly canonical trace/host bridges", "daily intent/state/calendar controls",
            "statistical input/chart hashes", "HPI original/prepared input equivalence", "source compile and whitespace"],
        "residual_limits": [report_dict["proof_limit"], "No deployed broker/account/scheduler acceptance, authorization or allocation.",
            "Research selection and current-vintage data; historical execution costs remain qualified or blocked as displayed.",
            "Fixed original-dollar cost sensitivity is not an impact model or strategy rerun.",
            "Historical HPI trace recorder-source drift is retained explicitly; independent saved-path reconciliation does not recover missing recorder bytes."],
        "current_git_status": subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--short"], cwd=REPO_PATH, text=True)}
    (ROOT_PATH / "verification.json").write_text(json.dumps(verification_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    excluded_name_set = {"artifact_manifest.json"}
    artifact_path_list = [source_path for source_path in ROOT_PATH.rglob("*") if source_path.is_file()
        and source_path.name not in excluded_name_set
        and not any("pytest_temp" in part_str or part_str in {"__pycache__", ".pytest_cache"} for part_str in source_path.relative_to(ROOT_PATH).parts)]
    artifact_path_list.extend(source_file_list)
    manifest_list = [{"path": str(source_path.relative_to(REPO_PATH)).replace("\\", "/"),
        "bytes": source_path.stat().st_size, "sha256": hash_file(source_path)} for source_path in sorted(set(artifact_path_list))]
    (ROOT_PATH / "artifact_manifest.json").write_text(json.dumps({"generated_utc": datetime.now(UTC).isoformat(),
        "head": head_str, "files": manifest_list,
        "note": "Archive includes superseded attempts; only report.tests.xml and linked final replay receipts qualify signoff. Manifest excludes itself and temporary test folders."}, indent=2), encoding="utf-8")
    verify_production_sources(frozen_dict)
    print(json.dumps({"sealed": True, "strategies": 17, "tests": verification_dict["selected_unique_tests"],
        "replays": replay_count_dict, "artifact_count": len(manifest_list)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
