import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts.research import run_spy_adaptive_momentum_regime_study as study_module
from scripts.research.run_spy_adaptive_momentum_regime_study import (
    SOURCE_PART_1_PATH,
    build_path_df,
    build_timing_return_df,
    parse_args,
    performance_metrics_dict,
)


class SpyAdaptiveMomentumRegimeStudyTests(unittest.TestCase):
    def test_source_paths_are_portable_and_can_be_overridden(self):
        self.assertNotIn(".codex\\attachments", str(SOURCE_PART_1_PATH).lower())
        self.assertNotIn("\\downloads\\", str(SOURCE_PART_1_PATH).lower())

        source_part_1_path = Path("input") / "part1.pdf"
        source_part_2_path = Path("input") / "part2.pdf"
        pasted_note_path = Path("input") / "note.txt"
        args_obj = parse_args(
            [
                "--write-contract",
                "--source-part-1-path",
                str(source_part_1_path),
                "--source-part-2-path",
                str(source_part_2_path),
                "--pasted-note-path",
                str(pasted_note_path),
            ]
        )

        self.assertEqual(args_obj.source_part_1_path, source_part_1_path)
        self.assertEqual(args_obj.source_part_2_path, source_part_2_path)
        self.assertEqual(args_obj.pasted_note_path, pasted_note_path)

    def test_missing_default_source_paths_are_reported_as_required(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            missing_root_path = Path(temp_dir_str) / "missing"
            with mock.patch.object(
                study_module,
                "SOURCE_PART_1_PATH",
                missing_root_path / "part1.pdf",
            ), mock.patch.object(
                study_module,
                "SOURCE_PART_2_PATH",
                missing_root_path / "part2.pdf",
            ), mock.patch.object(
                study_module,
                "PASTED_NOTE_PATH",
                missing_root_path / "note.txt",
            ):
                with self.assertRaisesRegex(FileNotFoundError, "source files are required"):
                    study_module.validate_source_files_and_hashes()

    def test_write_contract_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path = Path(temp_dir_str)
            source_part_1_path = temp_dir_path / "part1.pdf"
            source_part_2_path = temp_dir_path / "part2.pdf"
            pasted_note_path = temp_dir_path / "note.txt"
            source_part_1_path.write_bytes(b"part one")
            source_part_2_path.write_bytes(b"part two")
            pasted_note_path.write_text("commentary", encoding="utf-8")
            metadata_path = temp_dir_path / "data_snapshot.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "source_sha256": {
                            "part_1": "wrong",
                            "part_2": "wrong",
                            "pasted_note": "wrong",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                study_module, "SOURCE_PART_1_PATH", source_part_1_path
            ), mock.patch.object(
                study_module, "SOURCE_PART_2_PATH", source_part_2_path
            ), mock.patch.object(
                study_module, "PASTED_NOTE_PATH", pasted_note_path
            ), mock.patch.object(
                study_module, "DATA_METADATA_PATH", metadata_path
            ):
                with self.assertRaisesRegex(RuntimeError, "source hashes do not match"):
                    study_module.write_frozen_contract()

    def make_cache_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        return pd.DataFrame(
            {
                "spy_total_return_open": [100.0, 101.0, 103.0, 102.0, 106.0],
                "spy_total_return_close": [100.5, 102.0, 102.5, 105.0, 107.0],
            },
            index=date_index,
            dtype=float,
        )

    def test_timing_returns_match_declared_future_prices(self):
        cache_df = self.make_cache_df()
        timing_return_df = build_timing_return_df(cache_df)

        self.assertAlmostEqual(
            float(timing_return_df.iloc[0]["source_close_to_close"]),
            102.0 / 100.5 - 1.0,
        )
        self.assertAlmostEqual(
            float(timing_return_df.iloc[0]["pre_fill_overnight_to_next_open"]),
            101.0 / 100.5 - 1.0,
        )
        self.assertAlmostEqual(
            float(timing_return_df.iloc[0]["same_exit_intraday"]),
            102.0 / 101.0 - 1.0,
        )
        self.assertAlmostEqual(
            float(timing_return_df.iloc[0]["primary_next_open"]),
            103.0 / 101.0 - 1.0,
        )
        self.assertAlmostEqual(
            float(timing_return_df.iloc[0]["held_overnight_to_second_open"]),
            103.0 / 102.0 - 1.0,
        )
        source_compounded_float = (
            (1.0 + timing_return_df.iloc[0]["pre_fill_overnight_to_next_open"])
            * (1.0 + timing_return_df.iloc[0]["same_exit_intraday"])
            - 1.0
        )
        self.assertAlmostEqual(
            source_compounded_float,
            float(timing_return_df.iloc[0]["source_close_to_close"]),
        )
        executable_compounded_float = (
            (1.0 + timing_return_df.iloc[0]["same_exit_intraday"])
            * (1.0 + timing_return_df.iloc[0]["held_overnight_to_second_open"])
            - 1.0
        )
        self.assertAlmostEqual(
            executable_compounded_float,
            float(timing_return_df.iloc[0]["primary_next_open"]),
        )

    def test_cost_is_charged_only_on_changed_target_notional(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        target_weight_ser = pd.Series([0.0, 1.0, 1.0, 0.0], index=date_index)
        asset_return_ser = pd.Series([0.01, 0.02, -0.01, 0.03], index=date_index)
        path_df = build_path_df(
            target_weight_ser=target_weight_ser,
            asset_return_ser=asset_return_ser,
            round_trip_bps_float=10.0,
        )

        expected_cost_vec = np.array([0.0, 0.0005, 0.0, 0.0005])
        np.testing.assert_allclose(path_df["cost"].to_numpy(), expected_cost_vec)
        np.testing.assert_allclose(
            path_df["strategy_return"].to_numpy(),
            np.array([0.0, 0.0195, -0.01, -0.0005]),
        )

    def test_metrics_report_exposure_turnover_beta_and_drawdown(self):
        date_index = pd.date_range("2024-01-02", periods=6, freq="B")
        target_weight_ser = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], index=date_index)
        asset_return_ser = pd.Series([0.01, 0.02, -0.10, 0.03, 0.04, 0.01], index=date_index)
        path_df = build_path_df(target_weight_ser, asset_return_ser, 0.0)
        metric_dict = performance_metrics_dict(path_df)

        self.assertAlmostEqual(float(metric_dict["average_exposure"]), 4.0 / 6.0)
        self.assertEqual(int(metric_dict["state_change_count"]), 3)
        self.assertEqual(int(metric_dict["entry_count"]), 2)
        self.assertLess(float(metric_dict["maximum_drawdown"]), 0.0)
        self.assertTrue(np.isfinite(float(metric_dict["market_beta"])))


if __name__ == "__main__":
    unittest.main()
