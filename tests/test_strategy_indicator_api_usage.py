import re
import unittest
from pathlib import Path


WORKSPACE_PATH = Path(__file__).resolve().parents[1]
STRATEGY_DIR_PATH = WORKSPACE_PATH / "strategies"


class StrategyIndicatorApiUsageTests(unittest.TestCase):
    def test_strategy_scripts_using_shared_indicators_import_shared_api(self):
        strategy_file_path_list = sorted(STRATEGY_DIR_PATH.rglob("*.py"))
        strategy_file_path_list = [
            strategy_file_path
            for strategy_file_path in strategy_file_path_list
            if strategy_file_path.name != "run_portfolio.py"
        ]

        missing_import_file_list: list[str] = []
        for strategy_file_path in strategy_file_path_list:
            file_text_str = strategy_file_path.read_text(encoding="utf-8")
            uses_shared_indicator_bool = any(
                indicator_name_str in file_text_str
                for indicator_name_str in [
                    "dv2_indicator(",
                    "qp_indicator(",
                    "adv_dollar_indicator(",
                    "ibs_indicator(",
                ]
            )
            if uses_shared_indicator_bool and "from alpha.indicators import" not in file_text_str:
                missing_import_file_list.append(str(strategy_file_path))

        self.assertEqual(
            missing_import_file_list,
            [],
            msg=f"Strategy files missing alpha.indicators import: {missing_import_file_list}",
        )

    def test_strategy_scripts_do_not_use_legacy_indicator_imports(self):
        strategy_file_path_list = sorted(STRATEGY_DIR_PATH.rglob("*.py"))

        legacy_usage_list: list[str] = []
        for strategy_file_path in strategy_file_path_list:
            file_text_str = strategy_file_path.read_text(encoding="utf-8")
            if "from alpha.engine.indicators import" in file_text_str:
                legacy_usage_list.append(str(strategy_file_path))

        self.assertEqual(
            legacy_usage_list,
            [],
            msg=f"Legacy alpha.engine.indicators imports found in: {legacy_usage_list}",
        )

    def test_strategy_scripts_do_not_import_engine_fast_indicator_modules_directly(self):
        strategy_file_path_list = sorted(STRATEGY_DIR_PATH.rglob("*.py"))
        forbidden_pattern_list = [
            r"alpha\.engine\.dv2_indicator_fast",
            r"alpha\.engine\.qp_indicator_fast",
        ]

        forbidden_usage_list: list[str] = []
        for strategy_file_path in strategy_file_path_list:
            file_text_str = strategy_file_path.read_text(encoding="utf-8")
            if any(re.search(pattern_str, file_text_str) for pattern_str in forbidden_pattern_list):
                forbidden_usage_list.append(str(strategy_file_path))

        self.assertEqual(
            forbidden_usage_list,
            [],
            msg=f"Direct engine fast-indicator imports found in: {forbidden_usage_list}",
        )


if __name__ == "__main__":
    unittest.main()
