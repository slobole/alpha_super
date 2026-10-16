"""Refresh the MR technical review from completed isolated evidence only."""
import json
from pathlib import Path

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
AUDIT_PATH_OBJ = REPO_PATH_OBJ / "results/review/wired_signoff_20261015"


def main():
    review_path_obj = AUDIT_PATH_OBJ / "mr_review.json"
    review_dict = json.loads(review_path_obj.read_text(encoding="utf-8"))
    replay_by_variant_dict = {}
    for replay_path_obj in sorted(AUDIT_PATH_OBJ.glob("mr_*_replay_v*/replay.json")):
        replay_dict = json.loads(replay_path_obj.read_text(encoding="utf-8"))
        if replay_dict.get("status_str") != "pass" or replay_dict.get("coverage_dict", {}).get("with_order_intent", 0) < 20:
            continue
        row_list = replay_dict["row_list"]
        if not replay_dict.get("source_unchanged_bool") or any(
            row_dict.get("strategy_state_equal_bool") is not True or
            not row_dict.get("signal_close_timestamp_equal_bool") or
            not row_dict.get("execution_timestamp_equal_bool") or
            not row_dict.get("wrong_execution_date_positive_control_caught_bool")
            for row_dict in row_list
        ):
            continue
        replay_by_variant_dict[replay_dict["variant_str"]] = {
            "variant": replay_dict["variant_str"], "strategy_import_str": replay_dict["strategy_import_str"],
            "artifact": str(replay_path_obj.relative_to(AUDIT_PATH_OBJ)).replace("\\", "/"),
            "sessions": replay_dict["session_count_int"], "passed": replay_dict["passed_session_count_int"],
            "coverage": replay_dict["coverage_dict"], "source_hashes_unchanged": replay_dict["source_unchanged_bool"],
            "post_iterate_state_compared": sum(row_dict.get("strategy_state_equal_bool") is not None for row_dict in row_list),
            "timestamp_gates_passed": sum(row_dict.get("signal_close_timestamp_equal_bool") and row_dict.get("execution_timestamp_equal_bool") for row_dict in row_list),
            "wrong_execution_date_controls_caught": sum(row_dict.get("wrong_execution_date_positive_control_caught_bool", False) for row_dict in row_list),
        }
    review_dict["native_replays_completed"] = list(replay_by_variant_dict.values())
    review_dict["all_nine_native_host_replays_complete_bool"] = len(replay_by_variant_dict) == 9
    if len(replay_by_variant_dict) == 9:
        review_dict["status"] = "offline_audit_complete_with_open_findings"
    capacity_path_obj = AUDIT_PATH_OBJ / "mr_capacity_screen_v3.json"
    if capacity_path_obj.exists():
        review_dict["capacity_screen_artifact"] = capacity_path_obj.name
        review_dict["capacity_screen_strategy_count"] = len(json.loads(capacity_path_obj.read_text(encoding="utf-8"))["strategy_list"])
    review_path_obj.write_text(json.dumps(review_dict, indent=2, default=str), encoding="utf-8")
    print(f"Completed native host replays: {len(replay_by_variant_dict)}/9")


if __name__ == "__main__":
    main()
