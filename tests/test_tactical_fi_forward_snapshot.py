from datetime import UTC, datetime
from pathlib import Path
import sys
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from scripts.research import tactical_fi_forward_snapshot as forward_module


class FakeResponse:
    def __init__(self, csv_bytes: bytes) -> None:
        self.csv_bytes = csv_bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        return False

    def read(self) -> bytes:
        return self.csv_bytes


def test_capture_writes_four_hash_checked_inputs(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(source_url_str: str, timeout: int) -> FakeResponse:
        assert timeout == 30
        series_id_str = source_url_str.split("=")[-1]
        assert series_id_str in forward_module.tactical_module.FRED_SERIES_ID_TUPLE
        return FakeResponse(
            f"DATE,{series_id_str}\n2026-08-17,4.00\n2026-08-18,4.10\n".encode()
        )

    monkeypatch.setattr(forward_module, "urlopen", fake_urlopen)
    snapshot_path = forward_module.capture_snapshot(tmp_path)
    yield_df, manifest_dict = forward_module.load_snapshot(snapshot_path)

    assert list(yield_df.columns) == list(forward_module.tactical_module.FRED_SERIES_ID_TUPLE)
    assert yield_df.loc["2026-08-18", "DGS10"] == pytest.approx(4.10)
    assert set(manifest_dict["series_by_id_dict"]) == set(yield_df.columns)
    assert all(
        manifest_dict["series_by_id_dict"][series_id_str]["captured_at_utc_str"]
        for series_id_str in yield_df.columns
    )

    (snapshot_path / "DGS10.csv").write_text("DATE,DGS10\n2026-08-18,5.00\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        forward_module.load_snapshot(snapshot_path)


def test_capture_refuses_partial_download_without_cache(tmp_path: Path, monkeypatch) -> None:
    def failing_urlopen(source_url_str: str, timeout: int) -> FakeResponse:
        if "DGS3MO" in source_url_str:
            raise URLError("unavailable")
        return FakeResponse(b"DATE,DGS10\n2026-08-18,4.00\n")

    monkeypatch.setattr(forward_module, "urlopen", failing_urlopen)
    with pytest.raises(URLError):
        forward_module.capture_snapshot(tmp_path)
    assert not tmp_path.joinpath("manifest.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_parse_refuses_duplicate_observation_dates() -> None:
    with pytest.raises(ValueError, match="unique"):
        forward_module.parse_fred_series(
            "DGS10", b"DATE,DGS10\n2026-08-18,4.00\n2026-08-18,4.10\n"
        )


@pytest.mark.parametrize("bad_value_str", ["revised?", "Infinity"])
def test_parse_refuses_invalid_fred_numeric_values(bad_value_str: str) -> None:
    with pytest.raises(ValueError, match="invalid numeric value"):
        forward_module.parse_fred_series(
            "DGS10", f"DATE,DGS10\n2026-08-18,{bad_value_str}\n".encode()
        )
    missing_ser = forward_module.parse_fred_series(
        "DGS10", b"DATE,DGS10\n2026-08-18,.\n2026-08-19,4.0\n"
    )
    assert missing_ser.index[-1] == pd.Timestamp("2026-08-19")
    blank_ser = forward_module.parse_fred_series(
        "DGS10", b"DATE,DGS10\n2026-08-18,\n2026-08-19,4.0\n"
    )
    assert blank_ser.index[-1] == pd.Timestamp("2026-08-19")


def test_capture_timing_requires_month_end_after_close_and_before_cutoff() -> None:
    manifest_dict = {
        "series_by_id_dict": {
            series_id_str: {"captured_at_utc_str": "2026-09-30T21:05:00+00:00"}
            for series_id_str in forward_module.tactical_module.FRED_SERIES_ID_TUPLE
        }
    }
    assert forward_module.assess_capture_timing(manifest_dict) == {
        "source_received_in_decision_window_bool": True,
        "reason_code_str": "eligible",
        "market_date_str": "2026-09-30",
    }

    manifest_dict["series_by_id_dict"]["DGS10"]["captured_at_utc_str"] = (
        "2026-09-30T19:59:00+00:00"
    )
    assert forward_module.assess_capture_timing(manifest_dict)["reason_code_str"] == (
        "capture_before_market_close"
    )

    manifest_dict["series_by_id_dict"]["DGS10"]["captured_at_utc_str"] = (
        "2026-09-30T21:16:00+00:00"
    )
    assert forward_module.assess_capture_timing(manifest_dict)["reason_code_str"] == (
        "capture_after_1715_et_cutoff"
    )


def test_forward_target_preserves_inclusive_median_and_strict_comparison() -> None:
    term_history_list = [1.0, 3.0]
    credit_history_list = [1.0, 3.0]
    yield_row_ser = pd.Series({"DGS10": 5.0, "DGS3MO": 3.0, "DAAA": 6.0, "DBAA": 8.0})

    target_dict = forward_module.build_forward_target(
        yield_row_ser, term_history_list, credit_history_list
    )

    assert target_dict["term_median_float"] == pytest.approx(2.0)
    assert target_dict["credit_median_float"] == pytest.approx(3.0)
    assert (target_dict["IEF_weight_float"], target_dict["LQD_weight_float"], target_dict["Cash_weight_float"]) == (
        0.0, 0.5, 0.5
    )
    assert term_history_list == [1.0, 3.0, 2.0]


def test_forward_history_keeps_frozen_prehistory_in_medians() -> None:
    frozen_yield_df, frozen_signal_df = (
        forward_module.load_governed_frozen_signal_history()
    )
    term_history_list, credit_history_list = forward_module.frozen_spread_history_lists(
        frozen_yield_df, frozen_signal_df
    )

    assert len(term_history_list) > len(frozen_signal_df)
    assert len(credit_history_list) > len(frozen_signal_df)
    assert np.median(term_history_list) == pytest.approx(
        frozen_signal_df["term_threshold_float"].iloc[-1]
    )
    assert np.median(credit_history_list) == pytest.approx(
        frozen_signal_df["credit_threshold_float"].iloc[-1]
    )


def test_forward_decision_refuses_missing_august_snapshot(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(source_url_str: str, timeout: int) -> FakeResponse:
        series_id_str = source_url_str.split("=")[-1]
        return FakeResponse(f"DATE,{series_id_str}\n2026-08-18,4.00\n".encode())

    monkeypatch.setattr(forward_module, "urlopen", fake_urlopen)
    snapshot_path = forward_module.capture_snapshot(tmp_path)
    manifest_path = snapshot_path / "manifest.json"
    manifest_dict = forward_module.json.loads(manifest_path.read_text(encoding="utf-8"))
    for series_metadata_dict in manifest_dict["series_by_id_dict"].values():
        series_metadata_dict["captured_at_utc_str"] = "2026-09-30T21:05:00+00:00"
    manifest_path.write_text(forward_module.json.dumps(manifest_dict), encoding="utf-8")

    report_dict = forward_module.build_forward_decision_report(tmp_path)

    assert report_dict["status_str"] == "missing_contiguous_month_end_snapshot"
    assert report_dict["missing_month_list"] == ["2026-08"]
    assert report_dict["decision_row_list"] == []


def test_forward_decision_uses_frozen_history_plus_one_august_row(
    tmp_path: Path, monkeypatch
) -> None:
    yield_value_by_id_dict = {"DGS10": 4.0, "DGS3MO": 3.0, "DAAA": 5.0, "DBAA": 6.0}

    def fake_urlopen(source_url_str: str, timeout: int) -> FakeResponse:
        series_id_str = source_url_str.split("=")[-1]
        return FakeResponse(
            f"DATE,{series_id_str}\n2026-08-28,{yield_value_by_id_dict[series_id_str]}\n".encode()
        )

    monkeypatch.setattr(forward_module, "urlopen", fake_urlopen)
    snapshot_path = forward_module.capture_snapshot(tmp_path)
    manifest_path = snapshot_path / "manifest.json"
    manifest_dict = forward_module.json.loads(manifest_path.read_text(encoding="utf-8"))
    for series_metadata_dict in manifest_dict["series_by_id_dict"].values():
        series_metadata_dict["captured_at_utc_str"] = "2026-08-31T21:05:00+00:00"
    manifest_path.write_text(forward_module.json.dumps(manifest_dict), encoding="utf-8")

    report_dict = forward_module.build_forward_decision_report(tmp_path)
    decision_row_dict = report_dict["decision_row_list"][0]
    frozen_data_tuple = forward_module.tactical_module.get_tactical_yield_data(
        forward_module.tactical_module.DEFAULT_CONFIG
    )
    term_history_list, credit_history_list = forward_module.frozen_spread_history_lists(
        frozen_data_tuple[1], frozen_data_tuple[2]
    )
    expected_target_dict = forward_module.build_forward_target(
        pd.Series(yield_value_by_id_dict), term_history_list, credit_history_list
    )

    assert report_dict["status_str"] == "shadow_target_reconstructed_after_cutoff"
    assert decision_row_dict["decision_date_str"] == "2026-08-31"
    assert decision_row_dict["observation_date_str"] == "2026-08-28"
    assert decision_row_dict["next_open_date_str"] == "2026-09-01"
    assert decision_row_dict["observation_age_sessions_int"] == 0
    assert decision_row_dict["computed_by_1715_et_bool"] is False
    assert sum(decision_row_dict[f"{asset_str}_weight_float"] for asset_str in ("IEF", "LQD", "Cash")) == 1.0
    for target_name_str, expected_float in expected_target_dict.items():
        assert decision_row_dict[target_name_str] == pytest.approx(expected_float)


@pytest.mark.parametrize(
    ("status_str", "decision_row_list", "expected_exit_int"),
    [
        ("no_eligible_month_end_snapshot", [], 2),
        ("missing_contiguous_month_end_snapshot", [], 2),
        (
            "shadow_target_reconstructed_after_cutoff",
            [{"computed_by_1715_et_bool": False}],
            2,
        ),
        (
            "shadow_target_computed_by_cutoff",
            [
                {"computed_by_1715_et_bool": False},
                {"computed_by_1715_et_bool": True},
            ],
            2,
        ),
        (
            "shadow_target_computed_by_cutoff",
            [{"computed_by_1715_et_bool": True}],
            0,
        ),
    ],
)
def test_forward_command_fails_closed_when_decision_is_not_on_time(
    tmp_path: Path,
    monkeypatch,
    capsys,
    status_str: str,
    decision_row_list: list[dict],
    expected_exit_int: int,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 30, 21, 6, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(forward_module, "datetime", FixedDateTime)
    for decision_row_dict in decision_row_list:
        decision_row_dict["decision_date_str"] = "2026-09-30"
    monkeypatch.setattr(
        forward_module,
        "build_forward_decision_report",
        lambda snapshot_root_path: {
            "status_str": status_str,
            "decision_row_list": decision_row_list,
        },
    )
    monkeypatch.setattr(forward_module, "persist_on_time_decision", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tactical_fi_forward_snapshot.py", "forward", "--snapshot-root", str(tmp_path)],
    )

    assert forward_module.main() == expected_exit_int
    assert forward_module.json.loads(capsys.readouterr().out)["status_str"] == status_str


def test_forward_command_refuses_stale_recorded_month(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        forward_module,
        "build_forward_decision_report",
        lambda snapshot_root_path: {
            "status_str": "shadow_target_computed_by_cutoff",
            "decision_row_list": [{
                "decision_date_str": "2026-08-31",
                "computed_by_1715_et_bool": True,
            }],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["tactical_fi_forward_snapshot.py", "forward", "--snapshot-root", str(tmp_path)],
    )
    assert forward_module.main() == 2


def test_forward_decision_journal_preserves_original_months_and_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    class FixedDateTime(datetime):
        current_dt = datetime(2026, 8, 31, 21, 6, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current_dt.astimezone(tz)

    def fake_urlopen(source_url_str: str, timeout: int) -> FakeResponse:
        series_id_str = source_url_str.split("=")[-1]
        observation_date_str = (
            "2026-08-28" if FixedDateTime.current_dt.month == 8 else "2026-09-29"
        )
        return FakeResponse(
            f"DATE,{series_id_str}\n{observation_date_str},4.00\n".encode()
        )

    monkeypatch.setattr(forward_module, "datetime", FixedDateTime)
    monkeypatch.setattr(forward_module, "urlopen", fake_urlopen)
    august_snapshot_path = forward_module.capture_snapshot(tmp_path)
    august_report_dict = forward_module.build_forward_decision_report(tmp_path)
    assert august_report_dict["status_str"] == "shadow_target_computed_by_cutoff"
    forward_module.persist_on_time_decision(august_report_dict, tmp_path)

    FixedDateTime.current_dt = datetime(2026, 9, 30, 21, 6, tzinfo=UTC)
    september_snapshot_path = forward_module.capture_snapshot(tmp_path)
    september_report_dict = forward_module.build_forward_decision_report(tmp_path)
    assert september_report_dict["status_str"] == "shadow_target_computed_by_cutoff"
    assert all(
        decision_row_dict["computed_by_1715_et_bool"]
        for decision_row_dict in september_report_dict["decision_row_list"]
    )
    forward_module.persist_on_time_decision(september_report_dict, tmp_path)
    forward_module.persist_on_time_decision(september_report_dict, tmp_path)

    FixedDateTime.current_dt = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
    later_report_dict = forward_module.build_forward_decision_report(tmp_path)
    assert later_report_dict["decision_row_list"] == september_report_dict["decision_row_list"]
    assert later_report_dict["decision_row_list"][0]["snapshot_id_str"] == august_snapshot_path.name
    assert later_report_dict["decision_row_list"][1]["snapshot_id_str"] == september_snapshot_path.name

    record_path = tmp_path / "decision_records" / "2026-08.json"
    record_dict = forward_module.json.loads(record_path.read_text(encoding="utf-8"))
    record_dict["decision_row_dict"]["IEF_weight_float"] = 0.123
    record_path.write_text(forward_module.json.dumps(record_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded decision differs"):
        forward_module.build_forward_decision_report(tmp_path)
    with pytest.raises(ValueError, match="existing TFI decision record differs"):
        forward_module.persist_on_time_decision(august_report_dict, tmp_path)
