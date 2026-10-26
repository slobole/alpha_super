"""Issued monthly report content survives later source and mapping revisions."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.report_archive import (
    ARCHIVE_BYTE_LIMIT_INT,
    ReportArchiveError,
    ReportArchiveNotFoundError,
    ReportArchiveStore,
)


HTML_STR = '<!doctype html><html><style>body{font-family:serif}</style><body>September: $100</body></html>'


def _evidence_dict():
    return {"issued_at_str": "2026-10-05T12:00:00+00:00", "month_str": "2026-09",
        "source_hash_str": "source-v1", "program_by_pod_dict": {"pod_a": "Steady"}, "closing_value_str": "100.00"}


def _store_obj(tmp_path, *, memory_bool=False):
    return ReportArchiveStore(SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl")), memory_bool=memory_bool)


@pytest.mark.parametrize("memory_bool", [False, True])
def test_save_is_idempotent_and_read_returns_a_fresh_snapshot(tmp_path, memory_bool):
    store_obj = _store_obj(tmp_path, memory_bool=memory_bool)
    evidence_dict = _evidence_dict()
    archive_id_str = store_obj.save(HTML_STR, evidence_dict)
    assert len(archive_id_str) == 64
    assert store_obj.save(HTML_STR, dict(reversed(list(evidence_dict.items())))) == archive_id_str
    payload_dict = store_obj.read(archive_id_str)
    assert payload_dict == {"version_int": 1, "html_str": HTML_STR, "evidence_dict": evidence_dict}
    evidence_dict["program_by_pod_dict"]["pod_a"] = "Growth"
    payload_dict["evidence_dict"]["closing_value_str"] = "999.00"
    assert store_obj.read(archive_id_str)["evidence_dict"] == _evidence_dict()
    if not memory_bool:
        assert len(list((tmp_path / "monthly_reports").iterdir())) == 1
        assert _store_obj(tmp_path).read(archive_id_str) == store_obj.read(archive_id_str)
    else:
        assert not (tmp_path / "monthly_reports").exists()


def test_revisions_get_new_ids_without_replacing_previously_issued_reports(tmp_path):
    store_obj = _store_obj(tmp_path)
    first_id_str = store_obj.save(HTML_STR, _evidence_dict())
    first_bytes = (tmp_path / "monthly_reports" / (first_id_str + ".json")).read_bytes()
    revision_list = [
        (HTML_STR.replace("$100", "$105"), _evidence_dict()),
        (HTML_STR, {**_evidence_dict(), "source_hash_str": "source-v2"}),
        (HTML_STR, {**_evidence_dict(), "program_by_pod_dict": {"pod_a": "Growth"}}),
        (HTML_STR, {**_evidence_dict(), "issued_at_str": "2026-10-06T12:00:00+00:00"}),
    ]
    archive_id_set = {first_id_str}
    for html_str, evidence_dict in revision_list:
        archive_id_set.add(store_obj.save(html_str, evidence_dict))
    assert len(archive_id_set) == 5
    assert store_obj.read(first_id_str) == {"version_int": 1, "html_str": HTML_STR, "evidence_dict": _evidence_dict()}
    assert (tmp_path / "monthly_reports" / (first_id_str + ".json")).read_bytes() == first_bytes


def test_init_and_missing_reads_never_create_archive_directories(tmp_path):
    store_obj = _store_obj(tmp_path)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ReportArchiveNotFoundError):
        store_obj.read("0" * 64)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("archive_id_obj", ["../other", "A" * 64, "a" * 63, "a" * 65, "a" * 64 + "/", "", None])
def test_invalid_identifiers_cannot_escape_the_archive(tmp_path, archive_id_obj):
    store_obj = _store_obj(tmp_path)
    with pytest.raises(ReportArchiveError, match="identifier"):
        store_obj.read(archive_id_obj)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_existing_snapshot_is_never_overwritten_on_reissue(tmp_path):
    store_obj = _store_obj(tmp_path)
    archive_id_str = store_obj.save(HTML_STR, _evidence_dict())
    path_obj = tmp_path / "monthly_reports" / (archive_id_str + ".json")
    path_obj.write_bytes(b"corrupted snapshot")
    with pytest.raises(ReportArchiveError, match="integrity"):
        store_obj.read(archive_id_str)
    with pytest.raises(ReportArchiveError, match="integrity"):
        store_obj.save(HTML_STR, _evidence_dict())
    assert path_obj.read_bytes() == b"corrupted snapshot"


@pytest.mark.parametrize("payload_obj", [[], {}, {"version_int": 2, "html_str": HTML_STR, "evidence_dict": {}},
    {"version_int": True, "html_str": HTML_STR, "evidence_dict": {}},
    {"version_int": 1, "html_str": "", "evidence_dict": {}},
    {"version_int": 1, "html_str": HTML_STR, "evidence_dict": []}])
def test_matching_hash_does_not_bypass_payload_format_validation(tmp_path, payload_obj):
    directory_obj = tmp_path / "monthly_reports"
    directory_obj.mkdir()
    payload_bytes = json.dumps(payload_obj).encode("utf-8")
    archive_id_str = hashlib.sha256(payload_bytes).hexdigest()
    (directory_obj / (archive_id_str + ".json")).write_bytes(payload_bytes)
    with pytest.raises(ReportArchiveError, match="format"):
        _store_obj(tmp_path).read(archive_id_str)


def test_matching_hash_with_non_json_bytes_is_invalid(tmp_path):
    directory_obj = tmp_path / "monthly_reports"
    directory_obj.mkdir()
    payload_bytes = b"not JSON"
    archive_id_str = hashlib.sha256(payload_bytes).hexdigest()
    (directory_obj / (archive_id_str + ".json")).write_bytes(payload_bytes)
    with pytest.raises(ReportArchiveError, match="format"):
        _store_obj(tmp_path).read(archive_id_str)


def test_oversized_read_and_write_are_bounded(tmp_path):
    store_obj = _store_obj(tmp_path)
    with pytest.raises(ReportArchiveError, match="too large"):
        store_obj.save("x" * (ARCHIVE_BYTE_LIMIT_INT + 1), _evidence_dict())
    assert list(tmp_path.iterdir()) == []
    directory_obj = tmp_path / "monthly_reports"
    directory_obj.mkdir()
    payload_bytes = b"x" * (ARCHIVE_BYTE_LIMIT_INT + 1)
    archive_id_str = hashlib.sha256(payload_bytes).hexdigest()
    (directory_obj / (archive_id_str + ".json")).write_bytes(payload_bytes)
    with pytest.raises(ReportArchiveError, match="integrity"):
        store_obj.read(archive_id_str)


@pytest.mark.parametrize("html_obj,evidence_obj", [("", {}), (None, {}), (HTML_STR, []),
    (HTML_STR, {"bad": float("nan")}), (HTML_STR, {"bad": float("inf")}), (HTML_STR, {"bad": object()})])
def test_invalid_inputs_do_not_create_any_archive(tmp_path, html_obj, evidence_obj):
    with pytest.raises(ReportArchiveError):
        _store_obj(tmp_path).save(html_obj, evidence_obj)
    assert list(tmp_path.iterdir()) == []


def test_unconfigured_archive_location_does_not_fall_back_to_memory():
    store_obj = ReportArchiveStore(SimpleNamespace())
    with pytest.raises(ReportArchiveError, match="not configured"):
        store_obj.save(HTML_STR, _evidence_dict())
    demo_store_obj = ReportArchiveStore(SimpleNamespace(), memory_bool=True)
    archive_id_str = demo_store_obj.save(HTML_STR, _evidence_dict())
    assert demo_store_obj.read(archive_id_str)["html_str"] == HTML_STR


def test_permission_failure_is_reported_without_replacing_saved_reports(tmp_path, monkeypatch):
    store_obj = _store_obj(tmp_path)
    archive_id_str = store_obj.save(HTML_STR, _evidence_dict())
    before_bytes = (tmp_path / "monthly_reports" / (archive_id_str + ".json")).read_bytes()
    original_open_fn = Path.open

    def unavailable_obj(path_obj, mode_str="r", *argument_tuple, **keyword_dict):
        if path_obj.parent == tmp_path / "monthly_reports":
            raise PermissionError("Archive unavailable")
        return original_open_fn(path_obj, mode_str, *argument_tuple, **keyword_dict)

    monkeypatch.setattr(Path, "open", unavailable_obj)
    with pytest.raises(ReportArchiveError, match="could not be read"):
        store_obj.read(archive_id_str)
    with pytest.raises(ReportArchiveError, match="could not be archived"):
        store_obj.save(HTML_STR + " revised", _evidence_dict())
    monkeypatch.undo()
    assert (tmp_path / "monthly_reports" / (archive_id_str + ".json")).read_bytes() == before_bytes
    assert len(list((tmp_path / "monthly_reports").iterdir())) == 1


def test_partial_new_write_is_removed_and_can_be_retried(tmp_path, monkeypatch):
    store_obj = _store_obj(tmp_path)
    original_open_fn = Path.open

    class InterruptedWriter:
        def __init__(self, output_obj):
            self.output_obj = output_obj

        def __enter__(self):
            return self

        def __exit__(self, *argument_tuple):
            self.output_obj.close()

        def write(self, payload_bytes):
            self.output_obj.write(payload_bytes[:10])
            raise OSError("Write interrupted")

    def interrupted_obj(path_obj, mode_str="r", *argument_tuple, **keyword_dict):
        output_obj = original_open_fn(path_obj, mode_str, *argument_tuple, **keyword_dict)
        return InterruptedWriter(output_obj) if mode_str == "xb" else output_obj

    monkeypatch.setattr(Path, "open", interrupted_obj)
    with pytest.raises(ReportArchiveError, match="could not be archived"):
        store_obj.save(HTML_STR, _evidence_dict())
    assert list((tmp_path / "monthly_reports").iterdir()) == []
    monkeypatch.undo()
    archive_id_str = store_obj.save(HTML_STR, _evidence_dict())
    assert store_obj.read(archive_id_str)["html_str"] == HTML_STR
