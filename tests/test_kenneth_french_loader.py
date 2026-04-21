import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from alpha.data.kenneth_french_loader import (
    KennethFrenchSeriesUnavailableError,
    load_daily_kenneth_french_momentum_snapshot,
)


class _FakeUrlopenResponse:
    def __init__(self, response_bytes: bytes) -> None:
        self.response_bytes = response_bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        return False

    def read(self) -> bytes:
        return self.response_bytes


def build_test_zip_bytes(csv_text_str: str) -> bytes:
    buffer_obj = BytesIO()
    with ZipFile(buffer_obj, mode="w") as zip_file_obj:
        zip_file_obj.writestr("F-F_Momentum_Factor_daily.csv", csv_text_str)
    return buffer_obj.getvalue()


class KennethFrenchLoaderTests(unittest.TestCase):
    def test_load_daily_kenneth_french_momentum_snapshot_downloads_persists_and_parses_decimal_returns(self):
        csv_text_str = (
            "Kenneth French test file\n"
            ",Mom\n"
            "20070301,   1.25\n"
            "20070302,  -0.50\n"
            "20070305, -99.99\n"
            "Copyright 2026 Kenneth French\n"
        )
        zip_bytes = build_test_zip_bytes(csv_text_str)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_zip_path = Path(tmp_dir_str) / "F-F_Momentum_Factor_daily_CSV.zip"

            with patch(
                "alpha.data.kenneth_french_loader.urlopen",
                return_value=_FakeUrlopenResponse(zip_bytes),
            ) as urlopen_mock_obj:
                snapshot_obj = load_daily_kenneth_french_momentum_snapshot(
                    cache_zip_path_str=str(cache_zip_path),
                    as_of_ts=datetime(2007, 3, 2, tzinfo=UTC),
                    mode_str="backtest",
                )

            urlopen_mock_obj.assert_called_once()
            self.assertTrue(cache_zip_path.exists())
            self.assertEqual(cache_zip_path.read_bytes(), zip_bytes)
            self.assertEqual(snapshot_obj.download_status_str, "download_success")
            self.assertFalse(snapshot_obj.used_cache_bool)
            self.assertEqual(snapshot_obj.source_name_str, "Kenneth French Data Library")
            self.assertEqual(snapshot_obj.dataset_name_str, "F-F_Momentum_Factor_daily")
            self.assertEqual(snapshot_obj.latest_observation_date_ts, pd.Timestamp("2007-03-02"))
            self.assertEqual(list(snapshot_obj.value_ser.index), list(pd.to_datetime(["2007-03-01", "2007-03-02"])))
            self.assertAlmostEqual(float(snapshot_obj.value_ser.iloc[0]), 0.0125, places=12)
            self.assertAlmostEqual(float(snapshot_obj.value_ser.iloc[1]), -0.0050, places=12)

    def test_load_daily_kenneth_french_momentum_snapshot_falls_back_to_cache_on_download_failure(self):
        csv_text_str = ",Mom\n20070301,   1.25\n20070302,  -0.50\n"
        zip_bytes = build_test_zip_bytes(csv_text_str)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_zip_path = Path(tmp_dir_str) / "F-F_Momentum_Factor_daily_CSV.zip"
            cache_zip_path.write_bytes(zip_bytes)

            with patch(
                "alpha.data.kenneth_french_loader.urlopen",
                side_effect=URLError("network down"),
            ) as urlopen_mock_obj:
                snapshot_obj = load_daily_kenneth_french_momentum_snapshot(
                    cache_zip_path_str=str(cache_zip_path),
                    as_of_ts=datetime(2007, 3, 2, tzinfo=UTC),
                    mode_str="backtest",
                )

        urlopen_mock_obj.assert_called_once()
        self.assertEqual(snapshot_obj.download_status_str, "cache_fallback_after_download_error")
        self.assertTrue(snapshot_obj.used_cache_bool)
        self.assertEqual(snapshot_obj.latest_observation_date_ts, pd.Timestamp("2007-03-02"))
        self.assertAlmostEqual(float(snapshot_obj.value_ser.iloc[-1]), -0.0050, places=12)

    def test_load_daily_kenneth_french_momentum_snapshot_raises_when_download_and_cache_both_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_zip_path = Path(tmp_dir_str) / "F-F_Momentum_Factor_daily_CSV.zip"

            with patch(
                "alpha.data.kenneth_french_loader.urlopen",
                side_effect=URLError("network down"),
            ):
                with self.assertRaises(KennethFrenchSeriesUnavailableError) as exception_context:
                    load_daily_kenneth_french_momentum_snapshot(
                        cache_zip_path_str=str(cache_zip_path),
                        as_of_ts=datetime(2007, 3, 2, tzinfo=UTC),
                        mode_str="backtest",
                    )

        self.assertEqual(
            exception_context.exception.reason_code_str,
            "f-f_momentum_factor_daily_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
