import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

import pandas as pd

from alpha.data.fred_loader import (
    FredSeriesStaleError,
    FredSeriesUnavailableError,
    load_daily_fred_series_snapshot,
)


class _FakeUrlopenResponse:
    def __init__(self, response_text_str: str) -> None:
        self.response_text_str = response_text_str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        return False

    def read(self) -> bytes:
        return self.response_text_str.encode("utf-8")


class FredLoaderTests(unittest.TestCase):
    def test_load_daily_fred_series_snapshot_downloads_and_persists_cache(self):
        csv_text_str = "DATE,DTB3\n2024-01-31,5.00\n2024-02-01,5.10\n"

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_csv_path = Path(tmp_dir_str) / "DTB3.csv"

            with patch(
                "alpha.data.fred_loader.urlopen",
                return_value=_FakeUrlopenResponse(csv_text_str),
            ) as urlopen_mock_obj:
                snapshot_obj = load_daily_fred_series_snapshot(
                    series_id_str="DTB3",
                    cache_csv_path_str=str(cache_csv_path),
                    as_of_ts=datetime(2024, 2, 1, tzinfo=UTC),
                    mode_str="backtest",
                )

            urlopen_mock_obj.assert_called_once()
            self.assertTrue(cache_csv_path.exists())
            self.assertEqual(cache_csv_path.read_text(encoding="utf-8"), csv_text_str)
            self.assertEqual(snapshot_obj.download_status_str, "download_success")
            self.assertFalse(snapshot_obj.used_cache_bool)
            self.assertEqual(snapshot_obj.source_name_str, "FRED")
            self.assertEqual(snapshot_obj.series_id_str, "DTB3")
            self.assertEqual(snapshot_obj.latest_observation_date_ts, pd.Timestamp("2024-02-01"))
            self.assertEqual(snapshot_obj.freshness_business_days_int, 0)
            self.assertEqual(list(snapshot_obj.value_ser.index), list(pd.to_datetime(["2024-01-31", "2024-02-01"])))

    def test_load_daily_fred_series_snapshot_falls_back_to_cache_on_download_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_csv_path = Path(tmp_dir_str) / "DTB3.csv"
            cache_csv_path.write_text(
                "DATE,DTB3\n2024-02-01,5.20\n",
                encoding="utf-8",
            )

            with patch(
                "alpha.data.fred_loader.urlopen",
                side_effect=URLError("network down"),
            ) as urlopen_mock_obj:
                snapshot_obj = load_daily_fred_series_snapshot(
                    series_id_str="DTB3",
                    cache_csv_path_str=str(cache_csv_path),
                    as_of_ts=datetime(2024, 2, 1, tzinfo=UTC),
                    mode_str="backtest",
                )

        urlopen_mock_obj.assert_called_once()
        self.assertEqual(snapshot_obj.download_status_str, "cache_fallback_after_download_error")
        self.assertTrue(snapshot_obj.used_cache_bool)
        self.assertEqual(snapshot_obj.latest_observation_date_ts, pd.Timestamp("2024-02-01"))
        self.assertEqual(float(snapshot_obj.value_ser.iloc[-1]), 5.20)

    def test_load_daily_fred_series_snapshot_raises_when_download_and_cache_both_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_csv_path = Path(tmp_dir_str) / "DTB3.csv"

            with patch(
                "alpha.data.fred_loader.urlopen",
                side_effect=URLError("network down"),
            ):
                with self.assertRaises(FredSeriesUnavailableError) as exception_context:
                    load_daily_fred_series_snapshot(
                        series_id_str="DTB3",
                        cache_csv_path_str=str(cache_csv_path),
                        as_of_ts=datetime(2024, 2, 1, tzinfo=UTC),
                        mode_str="backtest",
                    )

        self.assertEqual(exception_context.exception.reason_code_str, "dtb3_unavailable")

    def test_load_daily_fred_series_snapshot_live_accepts_two_business_day_lag(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_csv_path = Path(tmp_dir_str) / "DTB3.csv"
            cache_csv_path.write_text(
                "DATE,DTB3\n2024-02-02,5.20\n",
                encoding="utf-8",
            )

            with patch(
                "alpha.data.fred_loader.urlopen",
                side_effect=URLError("network down"),
            ):
                snapshot_obj = load_daily_fred_series_snapshot(
                    series_id_str="DTB3",
                    cache_csv_path_str=str(cache_csv_path),
                    as_of_ts=datetime(2024, 2, 6, tzinfo=UTC),
                    mode_str="live",
                )

        self.assertEqual(snapshot_obj.freshness_business_days_int, 2)
        self.assertTrue(snapshot_obj.used_cache_bool)
        self.assertEqual(snapshot_obj.download_status_str, "cache_fallback_after_download_error")

    def test_load_daily_fred_series_snapshot_live_rejects_more_than_two_business_day_staleness(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            cache_csv_path = Path(tmp_dir_str) / "DTB3.csv"
            cache_csv_path.write_text(
                "DATE,DTB3\n2024-02-02,5.20\n",
                encoding="utf-8",
            )

            with patch(
                "alpha.data.fred_loader.urlopen",
                side_effect=URLError("network down"),
            ):
                with self.assertRaises(FredSeriesStaleError) as exception_context:
                    load_daily_fred_series_snapshot(
                        series_id_str="DTB3",
                        cache_csv_path_str=str(cache_csv_path),
                        as_of_ts=datetime(2024, 2, 7, tzinfo=UTC),
                        mode_str="live",
                    )

        self.assertEqual(exception_context.exception.reason_code_str, "dtb3_stale")
        self.assertIsNotNone(exception_context.exception.series_snapshot_obj)
        self.assertEqual(
            exception_context.exception.series_snapshot_obj.freshness_business_days_int,
            3,
        )


if __name__ == "__main__":
    unittest.main()
