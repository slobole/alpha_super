from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


EXCLUDED_RESULT_DIR_NAME_SET = frozenset(
    {
        'research',
        'live_reference_compare',
    }
)


def _is_legacy_dir_name(dir_name_str: str) -> bool:
    return dir_name_str.startswith('_legacy_')


def discover_archive_candidate_path_list(results_root_path: Path) -> list[Path]:
    if not results_root_path.exists():
        return []
    candidate_path_list: list[Path] = []
    for child_path in sorted(results_root_path.iterdir(), key=lambda path_obj: path_obj.name.lower()):
        if not child_path.is_dir():
            continue
        if child_path.name in EXCLUDED_RESULT_DIR_NAME_SET:
            continue
        if _is_legacy_dir_name(child_path.name):
            continue
        candidate_path_list.append(child_path)
    return candidate_path_list


def archive_research_results(
    results_root_path: Path,
    *,
    dry_run_bool: bool = False,
    timestamp_str: str | None = None,
) -> dict[str, object]:
    results_root_path = results_root_path.resolve()
    candidate_path_list = discover_archive_candidate_path_list(results_root_path)
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    archive_root_path = results_root_path / f'_legacy_research_{timestamp_str}'

    if not dry_run_bool and len(candidate_path_list) > 0:
        archive_root_path.mkdir(parents=True, exist_ok=False)
        for candidate_path in candidate_path_list:
            destination_path = archive_root_path / candidate_path.name
            shutil.move(str(candidate_path), str(destination_path))

    return {
        'results_root_path_str': str(results_root_path),
        'archive_root_path_str': str(archive_root_path),
        'dry_run_bool': bool(dry_run_bool),
        'moved_count_int': 0 if dry_run_bool else int(len(candidate_path_list)),
        'candidate_path_list': [str(candidate_path) for candidate_path in candidate_path_list],
    }


def main() -> int:
    parser_obj = argparse.ArgumentParser(
        description='Archive old top-level research result folders into a dated legacy folder.'
    )
    parser_obj.add_argument(
        '--results-root',
        default='results',
        help='Root results directory to archive.',
    )
    parser_obj.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would move without changing files.',
    )
    args_obj = parser_obj.parse_args()

    result_dict = archive_research_results(
        Path(args_obj.results_root),
        dry_run_bool=bool(args_obj.dry_run),
    )
    action_str = 'Would move' if result_dict['dry_run_bool'] else 'Moved'
    print(
        f"{action_str} {len(result_dict['candidate_path_list'])} folder(s) "
        f"to {result_dict['archive_root_path_str']}"
    )
    for candidate_path_str in result_dict['candidate_path_list']:
        print(candidate_path_str)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
