from pathlib import Path

from scripts.archive_research_results import archive_research_results


def test_archive_research_results_dry_run_excludes_live_and_existing_structural_dirs(tmp_path):
    results_root_path = tmp_path / "results"
    (results_root_path / "strategy_mr_qpi").mkdir(parents=True)
    (results_root_path / "CurrentBook").mkdir()
    (results_root_path / "research").mkdir()
    (results_root_path / "live_reference_compare").mkdir()
    (results_root_path / "_archive_07-05-2026").mkdir()
    (results_root_path / "_legacy_research_2026-05-01_120000").mkdir()

    result_dict = archive_research_results(
        results_root_path,
        dry_run_bool=True,
        timestamp_str="2026-05-07_120000",
    )

    candidate_name_set = {
        Path(candidate_path_str).name
        for candidate_path_str in result_dict["candidate_path_list"]
    }
    assert candidate_name_set == {"CurrentBook", "strategy_mr_qpi"}
    assert (results_root_path / "strategy_mr_qpi").exists()
    assert not (results_root_path / "_legacy_research_2026-05-07_120000").exists()


def test_archive_research_results_moves_only_old_top_level_research_dirs(tmp_path):
    results_root_path = tmp_path / "results"
    (results_root_path / "strategy_mr_qpi").mkdir(parents=True)
    (results_root_path / "live_reference_compare").mkdir()

    result_dict = archive_research_results(
        results_root_path,
        dry_run_bool=False,
        timestamp_str="2026-05-07_120000",
    )

    archive_root_path = Path(result_dict["archive_root_path_str"])
    assert result_dict["moved_count_int"] == 1
    assert (archive_root_path / "strategy_mr_qpi").exists()
    assert not (results_root_path / "strategy_mr_qpi").exists()
    assert (results_root_path / "live_reference_compare").exists()
