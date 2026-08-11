from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from alpha.live.ibkr_performance_sync import (
    FlexSyncError,
    _date_chunk_tuple_list,
    _eod_boundary_date_tuple,
    _latest_completed_session_date_str,
    bootstrap_remote,
    fetch_flex_statement_xml_str,
    import_xml_bool,
    sync_range_bool,
)
from alpha.live.ibkr_performance import (
    PerformanceContractError,
    PerformanceStore,
    PodPerformanceBinding,
    build_performance_page_dict,
)


class _FakeResponse:
    def __init__(self, response_text_str: str) -> None:
        self.response_bytes = response_text_str.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exception_type_obj, exception_obj, traceback_obj) -> None:
        return None

    def read(self) -> bytes:
        return self.response_bytes


def test_flex_fetch_uses_two_step_flow_and_polls_only_get_statement() -> None:
    response_text_str_list = [
        """
        <FlexStatementResponse>
          <Status>Success</Status><ReferenceCode>12345</ReferenceCode>
        </FlexStatementResponse>
        """,
        """
        <FlexStatementResponse>
          <Status>Fail</Status><ErrorCode>1019</ErrorCode>
          <ErrorMessage>Statement generation in progress.</ErrorMessage>
        </FlexStatementResponse>
        """,
        '<FlexQueryResponse queryName="ALPHA_DAILY_TWR"><FlexStatements count="0" /></FlexQueryResponse>',
    ]
    request_url_str_list: list[str] = []
    sleep_seconds_float_list: list[float] = []

    def fake_urlopen_fn(request_obj, timeout):
        request_url_str_list.append(request_obj.full_url)
        assert timeout == 30.0
        return _FakeResponse(response_text_str_list.pop(0))

    result_xml_text_str = fetch_flex_statement_xml_str(
        token_str="secret-token",
        query_id_str="9001",
        from_date_str="2026-08-01",
        to_date_str="2026-08-07",
        urlopen_fn=fake_urlopen_fn,
        sleep_fn=sleep_seconds_float_list.append,
    )

    assert result_xml_text_str.startswith("<FlexQueryResponse")
    assert len(request_url_str_list) == 3
    assert "/SendRequest?" in request_url_str_list[0]
    assert "fd=20260801" in request_url_str_list[0]
    assert "td=20260807" in request_url_str_list[0]
    assert "/GetStatement?" in request_url_str_list[1]
    assert "q=12345" in request_url_str_list[1]
    assert sleep_seconds_float_list == [10.0]


def test_network_error_does_not_echo_token() -> None:
    def failing_urlopen_fn(request_obj, timeout):
        raise URLError("network down")

    with pytest.raises(FlexSyncError) as exception_info_obj:
        fetch_flex_statement_xml_str(
            token_str="top-secret-token",
            query_id_str="9001",
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            urlopen_fn=failing_urlopen_fn,
            sleep_fn=lambda _: None,
        )

    assert "top-secret-token" not in str(exception_info_obj.value)
    assert "network request failed" in str(exception_info_obj.value)


def test_flex_service_error_does_not_echo_returned_secret() -> None:
    response_text_str = """
    <FlexStatementResponse>
      <Status>Fail</Status><ErrorCode>1012</ErrorCode>
      <ErrorMessage>Rejected token top-secret-token for query 9001.</ErrorMessage>
    </FlexStatementResponse>
    """

    with pytest.raises(FlexSyncError) as exception_info_obj:
        fetch_flex_statement_xml_str(
            token_str="top-secret-token",
            query_id_str="9001",
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            urlopen_fn=lambda request_obj, timeout: _FakeResponse(response_text_str),
            sleep_fn=lambda _: None,
        )

    error_str = str(exception_info_obj.value)
    assert "top-secret-token" not in error_str
    assert "9001" not in error_str
    assert "1012" in error_str


def test_failed_sync_is_durable_and_visible_without_erasing_rows(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    binding_obj = PodPerformanceBinding(
        "U100", "pod_a", "2026-08-07", None, True
    )
    xml_text_str = """
    <FlexQueryResponse queryName="ALPHA_DAILY_TWR" type="AF">
      <FlexStatements count="1"><FlexStatement accountId="U100">
        <AccountInformation accountId="U100" currency="USD" />
        <ChangeInNAV accountId="U100" currency="USD" fromDate="20260807"
          toDate="20260807" startingValue="100" endingValue="101" twr="1" />
      </FlexStatement></FlexStatements>
    </FlexQueryResponse>
    """
    store_obj = PerformanceStore(db_path_str)
    store_obj.replace_range(
        xml_text_str=xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=[binding_obj],
    )

    with pytest.raises(FlexSyncError):
        sync_range_bool(
            db_path_str=db_path_str,
            binding_obj_list=[binding_obj],
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            query_name_str="ALPHA_DAILY_TWR",
            token_str="top-secret-token",
            query_id_str="9001",
            urlopen_fn=lambda request_obj, timeout: (_ for _ in ()).throw(
                URLError("network down")
            ),
            sleep_fn=lambda _: None,
        )

    page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 10, 10, 0, tzinfo=UTC),
    )
    assert len(store_obj.load_daily_row_list()) == 1
    assert page_dict["status_str"] == "error"
    assert page_dict["status_label_str"] == "Last shadow sync failed"
    assert "top-secret-token" not in page_dict["detail_str"]
    assert page_dict["fund_chart_dict"] is not None


def test_local_xml_overlap_requires_explicit_replace(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    xml_path_obj = tmp_path / "performance.xml"
    binding_obj = PodPerformanceBinding(
        "U100", "pod_a", "2026-08-07", None, True
    )
    original_xml_text_str = """
    <FlexQueryResponse queryName="ALPHA_DAILY_TWR" type="AF">
      <FlexStatements count="1"><FlexStatement accountId="U100">
        <AccountInformation accountId="U100" currency="USD" />
        <ChangeInNAV accountId="U100" currency="USD" fromDate="20260807"
          toDate="20260807" startingValue="100" endingValue="101" twr="1" />
      </FlexStatement></FlexStatements>
    </FlexQueryResponse>
    """
    corrected_xml_text_str = original_xml_text_str.replace(
        'endingValue="101" twr="1"', 'endingValue="102" twr="2"'
    )
    xml_path_obj.write_text(original_xml_text_str, encoding="utf-8")
    assert import_xml_bool(
        db_path_str=db_path_str,
        xml_path_str=str(xml_path_obj),
        binding_obj_list=[binding_obj],
        query_name_str="ALPHA_DAILY_TWR",
    )
    xml_path_obj.write_text(corrected_xml_text_str, encoding="utf-8")

    with pytest.raises(FlexSyncError, match="overlaps existing"):
        import_xml_bool(
            db_path_str=db_path_str,
            xml_path_str=str(xml_path_obj),
            binding_obj_list=[binding_obj],
            query_name_str="ALPHA_DAILY_TWR",
        )


def test_latest_completed_session_is_always_d_plus_one() -> None:
    assert _latest_completed_session_date_str(
        datetime(2026, 8, 10, 10, 15, tzinfo=UTC)
    ) == "2026-08-07"
    assert _latest_completed_session_date_str(
        datetime(2026, 8, 10, 21, 0, tzinfo=UTC)
    ) == "2026-08-07"


def test_bootstrap_chunks_never_exceed_365_calendar_days() -> None:
    chunk_tuple_list = _date_chunk_tuple_list("2024-01-01", "2026-08-07")

    assert chunk_tuple_list[0] == ("2024-01-01", "2024-12-30")
    assert chunk_tuple_list[-1][1] == "2026-08-07"
    for chunk_from_date_str, chunk_to_date_str in chunk_tuple_list:
        chunk_day_count_int = (
            datetime.fromisoformat(chunk_to_date_str).date()
            - datetime.fromisoformat(chunk_from_date_str).date()
        ).days + 1
        assert chunk_day_count_int <= 365


def test_bootstrap_failure_closes_sqlite_and_preserves_original_error(
    tmp_path, monkeypatch
) -> None:
    db_path_obj = tmp_path / "performance.sqlite3"
    temporary_db_path_obj = tmp_path / "performance.sqlite3.bootstrap.tmp"
    temporary_sidecar_path_obj_list = [
        tmp_path / "performance.sqlite3.bootstrap.tmp-journal",
        tmp_path / "performance.sqlite3.bootstrap.tmp-wal",
        tmp_path / "performance.sqlite3.bootstrap.tmp-shm",
    ]
    for temporary_sidecar_path_obj in temporary_sidecar_path_obj_list:
        temporary_sidecar_path_obj.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.fetch_flex_statement_xml_str",
        lambda **_kwargs: "<not-valid-flex-xml>",
    )

    with pytest.raises(PerformanceContractError, match="Flex XML"):
        bootstrap_remote(
            db_path_str=str(db_path_obj),
            binding_obj_list=[
                PodPerformanceBinding("U100", "pod_a", "2026-08-07", None, True)
            ],
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            query_name_str="ALPHA_DAILY_TWR",
            token_str="secret-token",
            query_id_str="12345",
            replace_bool=False,
        )

    assert not db_path_obj.exists()
    assert not temporary_db_path_obj.exists()
    assert not any(
        temporary_sidecar_path_obj.exists()
        for temporary_sidecar_path_obj in temporary_sidecar_path_obj_list
    )


def test_bootstrap_cleanup_error_does_not_mask_original_flex_error(
    tmp_path, monkeypatch
) -> None:
    db_path_obj = tmp_path / "performance.sqlite3"
    original_unlink_fn = type(db_path_obj).unlink

    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.fetch_flex_statement_xml_str",
        lambda **_kwargs: "<not-valid-flex-xml>",
    )

    def _unlink_with_locked_temp(self, *args, **kwargs):
        if str(self).endswith(".bootstrap.tmp") and self.exists():
            raise PermissionError("simulated Windows file lock")
        return original_unlink_fn(self, *args, **kwargs)

    monkeypatch.setattr(type(db_path_obj), "unlink", _unlink_with_locked_temp)

    with pytest.raises(PerformanceContractError, match="Flex XML"):
        bootstrap_remote(
            db_path_str=str(db_path_obj),
            binding_obj_list=[
                PodPerformanceBinding("U100", "pod_a", "2026-08-07", None, True)
            ],
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            query_name_str="ALPHA_DAILY_TWR",
            token_str="secret-token",
            query_id_str="12345",
            replace_bool=False,
        )

    assert not db_path_obj.exists()


def test_bootstrap_fails_closed_when_stale_temp_cannot_be_removed(
    tmp_path, monkeypatch
) -> None:
    db_path_obj = tmp_path / "performance.sqlite3"
    temporary_db_path_obj = tmp_path / "performance.sqlite3.bootstrap.tmp"
    temporary_db_path_obj.write_text("stale", encoding="utf-8")
    original_unlink_fn = type(db_path_obj).unlink
    fetch_called_bool = False

    def _fetch_flex_statement_xml_str(**_kwargs):
        nonlocal fetch_called_bool
        fetch_called_bool = True
        return "<not-valid-flex-xml>"

    def _unlink_with_locked_temp(self, *args, **kwargs):
        if self == temporary_db_path_obj:
            raise PermissionError("simulated Windows file lock")
        return original_unlink_fn(self, *args, **kwargs)

    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.fetch_flex_statement_xml_str",
        _fetch_flex_statement_xml_str,
    )
    monkeypatch.setattr(type(db_path_obj), "unlink", _unlink_with_locked_temp)

    with pytest.raises(FlexSyncError, match="Cannot clean"):
        bootstrap_remote(
            db_path_str=str(db_path_obj),
            binding_obj_list=[
                PodPerformanceBinding("U100", "pod_a", "2026-08-07", None, True)
            ],
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            query_name_str="ALPHA_DAILY_TWR",
            token_str="secret-token",
            query_id_str="12345",
            replace_bool=False,
        )

    assert fetch_called_bool is False
    assert not db_path_obj.exists()


def test_bootstrap_refuses_a_second_concurrent_run(tmp_path) -> None:
    db_path_obj = tmp_path / "performance.sqlite3"
    lock_path_obj = tmp_path / "performance.sqlite3.bootstrap.lock"
    lock_path_obj.write_text("12345", encoding="utf-8")

    with pytest.raises(FlexSyncError, match="already running"):
        bootstrap_remote(
            db_path_str=str(db_path_obj),
            binding_obj_list=[
                PodPerformanceBinding("U100", "pod_a", "2026-08-07", None, True)
            ],
            from_date_str="2026-08-07",
            to_date_str="2026-08-07",
            query_name_str="ALPHA_DAILY_TWR",
            token_str="secret-token",
            query_id_str="12345",
            replace_bool=False,
        )

    assert not db_path_obj.exists()
    assert lock_path_obj.exists()


def test_only_broker_eod_rows_define_trusted_performance_boundary(
    tmp_path, monkeypatch
) -> None:
    db_path_obj = tmp_path / "pod.sqlite3"
    db_path_obj.touch()
    history_row_dict_list = [
        {
            "updated_timestamp_str": "2026-08-06T20:10:00+00:00",
            "snapshot_stage_str": "eod",
            "snapshot_source_str": "virtual_broker",
        },
        {
            "updated_timestamp_str": "2026-08-07T20:10:00+00:00",
            "snapshot_stage_str": "eod",
            "snapshot_source_str": "broker",
        },
    ]

    class _FakeStateStore:
        def __init__(self, db_path_str: str) -> None:
            assert db_path_str == str(db_path_obj)

        def get_pod_state_history_row_dict_list(self, pod_id_str: str):
            assert pod_id_str == "pod_a"
            return history_row_dict_list

    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.LiveStateStore", _FakeStateStore
    )
    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.runner._market_date_str_from_timestamp_str",
        lambda *, timestamp_str, release_obj: timestamp_str[:10],
    )

    boundary_tuple = _eod_boundary_date_tuple(
        SimpleNamespace(pod_id_str="pod_a"), str(db_path_obj)
    )

    assert boundary_tuple == ("2026-08-07", "2026-08-07")


def test_nonbroker_eod_does_not_create_trusted_performance_boundary(
    tmp_path, monkeypatch
) -> None:
    db_path_obj = tmp_path / "pod.sqlite3"
    db_path_obj.touch()

    class _FakeStateStore:
        def __init__(self, db_path_str: str) -> None:
            pass

        def get_pod_state_history_row_dict_list(self, pod_id_str: str):
            return [
                {
                    "updated_timestamp_str": "2026-08-07T20:10:00+00:00",
                    "snapshot_stage_str": "eod",
                    "snapshot_source_str": "pod_state",
                }
            ]

    monkeypatch.setattr(
        "alpha.live.ibkr_performance_sync.LiveStateStore", _FakeStateStore
    )

    assert _eod_boundary_date_tuple(
        SimpleNamespace(pod_id_str="pod_a"), str(db_path_obj)
    ) == (None, None)
