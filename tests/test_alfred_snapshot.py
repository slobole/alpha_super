import hashlib
import json
from urllib.error import URLError

import pandas as pd
import pytest

import alpha.data.alfred_snapshot as alfred_module
from alpha.data.alfred_snapshot import (
    ALFRED_SNAPSHOT_SCHEMA_STR,
    AlfredRequestError,
    AlfredSnapshotIntegrityError,
    AlfredVintageError,
    decode_vintage_value_ser,
    encode_vintage_run_df,
    fetch_alfred_vintage_dict,
    load_alfred_snapshot_manifest,
    load_alfred_vintage_snapshot,
    parse_alfred_vintage_csv,
    run_df_to_csv_bytes,
    sha256_bytes_str,
    verify_vintage_run_round_trip,
)


def _ser(value_by_date_dict: dict[str, float]) -> pd.Series:
    value_ser = pd.Series(
        list(value_by_date_dict.values()),
        index=pd.DatetimeIndex(pd.to_datetime(list(value_by_date_dict)), name="observation_date"),
        dtype=float,
        name="DAAA",
    )
    return value_ser.sort_index()


def test_parse_rejects_a_substituted_vintage_label() -> None:
    # ALFRED answers a pre-archive date with today's vintage under another label.
    csv_text_str = "observation_date,DAAA_20260928\n2014-03-31,4.5\n"

    with pytest.raises(AlfredVintageError, match="did not return vintage DAAA_20140401"):
        parse_alfred_vintage_csv(csv_text_str, "DAAA", [pd.Timestamp("2014-04-01")])


def test_parse_rejects_observation_after_vintage_date() -> None:
    csv_text_str = "observation_date,DAAA_20160104\n2016-01-04,4.0\n2016-01-05,4.1\n"

    with pytest.raises(AlfredVintageError, match="after its vintage date"):
        parse_alfred_vintage_csv(csv_text_str, "DAAA", [pd.Timestamp("2016-01-04")])


def test_parse_drops_missing_values_and_keeps_exact_columns() -> None:
    csv_text_str = (
        "observation_date,DAAA_20160105,DAAA_20160106\n"
        "2016-01-01,,\n"
        "2016-01-04,4.0,4.0\n"
        "2016-01-05,,4.2\n"
    )

    vintage_value_dict = parse_alfred_vintage_csv(
        csv_text_str,
        "DAAA",
        [pd.Timestamp("2016-01-05"), pd.Timestamp("2016-01-06")],
    )

    assert vintage_value_dict[pd.Timestamp("2016-01-05")].to_dict() == {
        pd.Timestamp("2016-01-04"): 4.0
    }
    assert vintage_value_dict[pd.Timestamp("2016-01-06")].to_dict() == {
        pd.Timestamp("2016-01-04"): 4.0,
        pd.Timestamp("2016-01-05"): 4.2,
    }


def _revision_vintage_dict() -> dict[pd.Timestamp, pd.Series]:
    # 01-04 is revised A -> B -> A; 01-05 disappears and returns; 01-06 is a
    # late backfill that first appears in the third vintage.
    return {
        pd.Timestamp("2016-01-07"): _ser({"2016-01-04": 4.0, "2016-01-05": 4.1}),
        pd.Timestamp("2016-01-08"): _ser({"2016-01-04": 4.5}),
        pd.Timestamp("2016-01-11"): _ser(
            {"2016-01-04": 4.0, "2016-01-05": 4.1, "2016-01-06": 4.2}
        ),
    }


def test_run_table_reproduces_every_sampled_vintage_exactly() -> None:
    vintage_value_dict = _revision_vintage_dict()

    run_df = encode_vintage_run_df(vintage_value_dict)

    verify_vintage_run_round_trip(run_df, vintage_value_dict, "DAAA")
    assert len(run_df) == 6
    as_of_first_ser = decode_vintage_value_ser(run_df, pd.Timestamp("2016-01-07"), "DAAA")
    # The backfilled 01-06 value and the later revision are invisible at 01-07.
    assert as_of_first_ser.to_dict() == {
        pd.Timestamp("2016-01-04"): 4.0,
        pd.Timestamp("2016-01-05"): 4.1,
    }


def test_round_trip_check_detects_a_corrupted_run_table() -> None:
    vintage_value_dict = _revision_vintage_dict()
    run_df = encode_vintage_run_df(vintage_value_dict)
    run_df.loc[run_df.index[0], "value"] = 9.9

    with pytest.raises(AlfredSnapshotIntegrityError, match="does not reproduce"):
        verify_vintage_run_round_trip(run_df, vintage_value_dict, "DAAA")


def _write_snapshot(tmp_path, vintage_value_dict) -> dict:
    run_df = encode_vintage_run_df(vintage_value_dict)
    csv_bytes = run_df_to_csv_bytes(run_df)
    (tmp_path / "alfred_daaa_runs.csv").write_bytes(csv_bytes)
    manifest_dict = {
        "schema_str": ALFRED_SNAPSHOT_SCHEMA_STR,
        "series_by_id_dict": {
            "DAAA": {
                "file_str": "alfred_daaa_runs.csv",
                "sha256_str": sha256_bytes_str(csv_bytes),
                "run_row_count_int": int(len(run_df)),
                "vintage_date_list": [
                    date_ts.strftime("%Y-%m-%d") for date_ts in sorted(vintage_value_dict)
                ],
            }
        },
    }
    manifest_bytes = json.dumps(manifest_dict, indent=2, sort_keys=True).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    return {"manifest_sha256_str": sha256_bytes_str(manifest_bytes)}


def test_stored_snapshot_answers_only_sampled_vintages(tmp_path) -> None:
    vintage_value_dict = _revision_vintage_dict()
    hash_dict = _write_snapshot(tmp_path, vintage_value_dict)
    manifest_dict = load_alfred_snapshot_manifest(tmp_path, hash_dict["manifest_sha256_str"])

    snapshot_obj = load_alfred_vintage_snapshot(tmp_path, "DAAA", manifest_dict)

    for vintage_date_ts, expected_value_ser in vintage_value_dict.items():
        pd.testing.assert_series_equal(
            snapshot_obj.value_ser_as_of(vintage_date_ts),
            expected_value_ser,
            check_freq=False,
        )
    with pytest.raises(AlfredSnapshotIntegrityError, match="was not sampled"):
        snapshot_obj.value_ser_as_of(pd.Timestamp("2016-01-09"))


def test_stored_snapshot_fails_loud_on_tampering(tmp_path) -> None:
    hash_dict = _write_snapshot(tmp_path, _revision_vintage_dict())

    with pytest.raises(AlfredSnapshotIntegrityError, match="manifest hash mismatch"):
        load_alfred_snapshot_manifest(tmp_path, "0" * 64)

    manifest_dict = load_alfred_snapshot_manifest(tmp_path, hash_dict["manifest_sha256_str"])
    run_path = tmp_path / "alfred_daaa_runs.csv"
    run_path.write_bytes(run_path.read_bytes().replace(b"4.1", b"4.3"))
    with pytest.raises(AlfredSnapshotIntegrityError, match="run file hash mismatch"):
        load_alfred_vintage_snapshot(tmp_path, "DAAA", manifest_dict)


def test_parse_rejects_repeated_observation_dates() -> None:
    csv_text_str = "observation_date,DAAA_20160105\n2016-01-04,4.0\n2016-01-04,4.1\n"

    with pytest.raises(AlfredVintageError, match="repeats observation dates"):
        parse_alfred_vintage_csv(csv_text_str, "DAAA", [pd.Timestamp("2016-01-05")])


def test_parse_accepts_an_observation_dated_on_the_vintage_date() -> None:
    csv_text_str = "observation_date,DAAA_20160104\n2016-01-01,3.9\n2016-01-04,4.0\n"

    vintage_value_dict = parse_alfred_vintage_csv(csv_text_str, "DAAA", [pd.Timestamp("2016-01-04")])

    assert vintage_value_dict[pd.Timestamp("2016-01-04")].index[-1] == pd.Timestamp("2016-01-04")


def test_loader_rejects_wrong_schema_and_row_count(tmp_path) -> None:
    _write_snapshot(tmp_path, _revision_vintage_dict())
    manifest_path = tmp_path / "manifest.json"
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest_path.write_text(json.dumps({**manifest_dict, "schema_str": "other"}), encoding="utf-8")
    with pytest.raises(AlfredSnapshotIntegrityError, match="Unsupported ALFRED snapshot schema"):
        load_alfred_snapshot_manifest(tmp_path, None)

    manifest_dict["series_by_id_dict"]["DAAA"]["run_row_count_int"] += 1
    with pytest.raises(AlfredSnapshotIntegrityError, match="row count mismatch"):
        load_alfred_vintage_snapshot(tmp_path, "DAAA", manifest_dict)
    with pytest.raises(AlfredSnapshotIntegrityError, match="not in the ALFRED snapshot"):
        load_alfred_vintage_snapshot(tmp_path, "DBAA", manifest_dict)


class _FakeResponse:
    def __init__(self, body_bytes: bytes) -> None:
        self.body_bytes = body_bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def read(self) -> bytes:
        return self.body_bytes


def _fake_body_bytes(url_str: str) -> bytes:
    vintage_str = url_str.split("vintage_date=")[1]
    date_list = vintage_str.split(",")
    header_str = ",".join(["observation_date"] + [f"DAAA_{d.replace('-', '')}" for d in date_list])
    return f"{header_str}\n2016-01-04,{','.join(['4.0'] * len(date_list))}\n".encode("utf-8")


def test_fetch_batches_requests_and_records_response_hashes(monkeypatch) -> None:
    url_list = []

    def fake_urlopen(request_obj, timeout):
        url_list.append(request_obj.full_url)
        return _FakeResponse(_fake_body_bytes(request_obj.full_url))

    monkeypatch.setattr(alfred_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(alfred_module.time, "sleep", lambda seconds_float: None)
    vintage_date_list = list(pd.bdate_range("2016-01-05", periods=5))

    vintage_value_dict, request_record_list = fetch_alfred_vintage_dict(
        "DAAA", vintage_date_list, batch_size_int=2
    )

    assert len(url_list) == 3
    assert sorted(vintage_value_dict) == vintage_date_list
    assert [record_dict["vintage_count_int"] for record_dict in request_record_list] == [2, 2, 1]
    assert request_record_list[0]["response_sha256_str"] == hashlib.sha256(
        _fake_body_bytes(url_list[0])
    ).hexdigest()


def test_fetch_retries_then_raises_request_error(monkeypatch) -> None:
    call_list = []

    def failing_urlopen(request_obj, timeout):
        call_list.append(1)
        raise URLError("offline")

    monkeypatch.setattr(alfred_module, "urlopen", failing_urlopen)
    monkeypatch.setattr(alfred_module.time, "sleep", lambda seconds_float: None)

    with pytest.raises(AlfredRequestError) as exception_info:
        fetch_alfred_vintage_dict("DAAA", [pd.Timestamp("2016-01-05")])
    assert len(call_list) == 4
    assert isinstance(exception_info.value.__cause__, URLError)
    # A network failure is not a "missing vintage": callers must not confuse them.
    assert not isinstance(exception_info.value, AlfredVintageError)


def test_fetch_recovers_after_one_failure(monkeypatch) -> None:
    call_list = []

    def flaky_urlopen(request_obj, timeout):
        call_list.append(1)
        if len(call_list) == 1:
            raise URLError("hiccup")
        return _FakeResponse(_fake_body_bytes(request_obj.full_url))

    monkeypatch.setattr(alfred_module, "urlopen", flaky_urlopen)
    monkeypatch.setattr(alfred_module.time, "sleep", lambda seconds_float: None)

    vintage_value_dict, _record_list = fetch_alfred_vintage_dict("DAAA", [pd.Timestamp("2016-01-05")])

    assert len(call_list) == 2
    assert vintage_value_dict[pd.Timestamp("2016-01-05")].tolist() == [4.0]


def test_fetch_rejects_a_substituted_vintage_in_the_response(monkeypatch) -> None:
    def substituting_urlopen(request_obj, timeout):
        return _FakeResponse(b"observation_date,DAAA_20260928\n2016-01-04,4.0\n")

    monkeypatch.setattr(alfred_module, "urlopen", substituting_urlopen)
    monkeypatch.setattr(alfred_module.time, "sleep", lambda seconds_float: None)

    with pytest.raises(AlfredVintageError, match="DAAA_20140401"):
        fetch_alfred_vintage_dict("DAAA", [pd.Timestamp("2014-04-01")])
