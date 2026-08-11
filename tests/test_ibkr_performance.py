from __future__ import annotations

from datetime import UTC, datetime
import math

import pytest

from alpha.live.ibkr_performance import (
    FlexDailyPerformance,
    PerformanceContractError,
    PerformanceStore,
    PodPerformanceBinding,
    build_fund_daily_performance_list,
    build_performance_page_dict,
    parse_flex_change_in_nav_xml,
)


def _binding_dict(
    *, start_date_str: str = "2026-08-07"
) -> dict[str, PodPerformanceBinding]:
    return {
        "U100": PodPerformanceBinding(
            account_route_str="U100",
            pod_id_str="pod_taa",
            return_start_date_str=start_date_str,
            return_end_date_str=None,
            enabled_bool=True,
        ),
        "U200": PodPerformanceBinding(
            account_route_str="U200",
            pod_id_str="pod_ndx",
            return_start_date_str=start_date_str,
            return_end_date_str=None,
            enabled_bool=True,
        ),
    }


def _xml_str(
    row_tuple_list: list[tuple[str, str, float, float, float]],
) -> str:
    statement_str_list: list[str] = []
    row_tuple_list_by_account_dict: dict[
        str, list[tuple[str, str, float, float, float]]
    ] = {}
    for row_tuple in row_tuple_list:
        row_tuple_list_by_account_dict.setdefault(row_tuple[0], []).append(row_tuple)
    for account_route_str, account_row_tuple_list in sorted(
        row_tuple_list_by_account_dict.items()
    ):
        change_in_nav_str = "".join(
            (
                f'<ChangeInNAV accountId="{account_route_str}" currency="USD" '
                f'fromDate="{market_date_str.replace("-", "")}" '
                f'toDate="{market_date_str.replace("-", "")}" '
                f'startingValue="{starting_nav_float}" '
                f'endingValue="{ending_nav_float}" twr="{twr_pct_float}" />'
            )
            for (
                _,
                market_date_str,
                starting_nav_float,
                ending_nav_float,
                twr_pct_float,
            ) in account_row_tuple_list
        )
        statement_str_list.append(
            f'<FlexStatement accountId="{account_route_str}">'
            f'<AccountInformation accountId="{account_route_str}" currency="USD" />'
            f"{change_in_nav_str}</FlexStatement>"
        )
    return (
        '<FlexQueryResponse queryName="ALPHA_DAILY_TWR" type="AF">'
        f'<FlexStatements count="{len(statement_str_list)}">'
        + "".join(statement_str_list)
        + "</FlexStatements></FlexQueryResponse>"
    )


def test_real_shaped_daily_report_matches_known_fund_composite() -> None:
    binding_by_account_route_dict = _binding_dict()
    xml_text_str = _xml_str(
        [
            ("U100", "2026-08-07", 19936.907075693, 20211.567075693, 1.377645986),
            ("U200", "2026-08-07", 12677.354573607, 12592.844573607, -0.666621727),
        ]
    )

    row_obj_list = parse_flex_change_in_nav_xml(
        xml_text_str,
        binding_by_account_route_dict,
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
    )
    fund_row_obj_list = build_fund_daily_performance_list(
        row_obj_list, binding_by_account_route_dict
    )

    assert len(row_obj_list) == 2
    assert row_obj_list[0].twr_float == pytest.approx(0.01377645986)
    assert fund_row_obj_list[0].fund_return_float == pytest.approx(
        0.00583027150, abs=1e-10
    )


def test_two_daily_returns_chain_to_fifteen_point_five_percent() -> None:
    binding_obj = PodPerformanceBinding("U100", "pod_taa", "2026-08-06", None, True)
    binding_by_account_route_dict = {"U100": binding_obj}
    row_obj_list = [
        FlexDailyPerformance(
            "U100", "pod_taa", "2026-08-06", 100.0, 110.0, 10.0, 0.10
        ),
        FlexDailyPerformance(
            "U100", "pod_taa", "2026-08-07", 110.0, 115.5, 5.0, 0.05
        ),
    ]

    fund_row_obj_list = build_fund_daily_performance_list(
        row_obj_list, binding_by_account_route_dict
    )
    cumulative_return_float = math.prod(
        1.0 + row_obj.fund_return_float for row_obj in fund_row_obj_list
    ) - 1.0

    assert cumulative_return_float == pytest.approx(0.155)


def test_fund_composite_withholds_flow_sensitive_day() -> None:
    binding_by_account_route_dict = _binding_dict()
    row_obj_list = [
        # U100 receives capital during the IBKR subperiod, but IBKR TWR says
        # the account made 10%. The adjusted base is therefore 220 / 1.10 = 200.
        FlexDailyPerformance("U100", "pod_taa", "2026-08-07", 100, 220, 10, 0.10),
        FlexDailyPerformance("U200", "pod_ndx", "2026-08-07", 100, 105, 5, 0.05),
    ]

    with pytest.raises(
        PerformanceContractError, match="combined history is unavailable"
    ):
        build_fund_daily_performance_list(
            row_obj_list, binding_by_account_route_dict
        )


def test_offsetting_intraday_flows_prove_combined_line_is_only_indicative() -> None:
    binding_by_account_route_dict = _binding_dict()
    row_obj_list = [
        # Deposit + withdrawal can offset in the daily NAV identity. The
        # account TWR is official, but the combined result cannot be Fund TWR.
        FlexDailyPerformance("U100", "pod_taa", "2026-08-07", 100, 110, 10, 0.10),
        FlexDailyPerformance("U200", "pod_ndx", "2026-08-07", 100, 100, 0, 0.0),
    ]

    indicative_row_obj = build_fund_daily_performance_list(
        row_obj_list, binding_by_account_route_dict
    )[0]

    assert indicative_row_obj.fund_return_float == pytest.approx(0.05)


def test_zero_holiday_rows_are_filtered_from_performance() -> None:
    binding_by_account_route_dict = _binding_dict(start_date_str="2026-07-02")
    xml_text_str = _xml_str(
        [
            ("U100", "2026-07-02", 100, 101, 1.0),
            ("U200", "2026-07-02", 100, 99, -1.0),
            ("U100", "2026-07-03", 101, 101, 0.0),
            ("U200", "2026-07-03", 99, 99, 0.0),
            ("U100", "2026-07-06", 101, 102.01, 1.0),
            ("U200", "2026-07-06", 99, 99.99, 1.0),
        ]
    )

    row_obj_list = parse_flex_change_in_nav_xml(
        xml_text_str,
        binding_by_account_route_dict,
        request_from_date_str="2026-07-02",
        request_to_date_str="2026-07-06",
    )

    assert {row_obj.market_date_str for row_obj in row_obj_list} == {
        "2026-07-02",
        "2026-07-06",
    }


def test_nonzero_holiday_row_fails_loud() -> None:
    xml_text_str = _xml_str(
        [
            ("U100", "2026-07-03", 100, 101, 1.0),
            ("U200", "2026-07-03", 100, 100, 0.0),
        ]
    )

    with pytest.raises(PerformanceContractError, match="non-session"):
        parse_flex_change_in_nav_xml(xml_text_str, _binding_dict(start_date_str="2026-07-03"))


def test_missing_account_fails_loud() -> None:
    xml_text_str = _xml_str(
        [("U100", "2026-08-07", 100, 101, 1.0)]
    )

    with pytest.raises(PerformanceContractError, match="missing configured account"):
        parse_flex_change_in_nav_xml(xml_text_str, _binding_dict())


def test_requested_session_omission_fails_loud() -> None:
    xml_text_str = _xml_str(
        [
            ("U100", "2026-08-06", 100, 101, 1.0),
            ("U200", "2026-08-06", 100, 101, 1.0),
        ]
    )

    with pytest.raises(PerformanceContractError, match="requested XNYS session 2026-08-07"):
        parse_flex_change_in_nav_xml(
            xml_text_str,
            _binding_dict(start_date_str="2026-08-06"),
            request_from_date_str="2026-08-06",
            request_to_date_str="2026-08-07",
        )


def test_store_is_idempotent_and_correction_replaces_range(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    binding_obj_list = list(_binding_dict().values())
    original_xml_text_str = _xml_str(
        [
            ("U100", "2026-08-07", 100, 101, 1.0),
            ("U200", "2026-08-07", 100, 99, -1.0),
        ]
    )
    store_obj = PerformanceStore(db_path_str)

    assert store_obj.replace_range(
        xml_text_str=original_xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=binding_obj_list,
    )
    assert not store_obj.replace_range(
        xml_text_str=original_xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=binding_obj_list,
    )
    corrected_xml_text_str = _xml_str(
        [
            ("U100", "2026-08-07", 100, 102, 2.0),
            ("U200", "2026-08-07", 100, 99, -1.0),
        ]
    )
    assert store_obj.replace_range(
        xml_text_str=corrected_xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=binding_obj_list,
    )

    row_obj_list = store_obj.load_daily_row_list()
    assert len(row_obj_list) == 2
    assert next(row_obj for row_obj in row_obj_list if row_obj.account_route_str == "U100").twr_float == pytest.approx(0.02)


def test_duplicate_checksum_still_rejects_account_to_pod_remap(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    store_obj = PerformanceStore(db_path_str)
    xml_text_str = _xml_str(
        [("U100", "2026-08-07", 100, 101, 1.0)]
    )
    original_binding_obj = PodPerformanceBinding(
        "U100", "pod_a", "2026-08-07", None, True
    )
    remapped_binding_obj = PodPerformanceBinding(
        "U100", "pod_b", "2026-08-07", None, True
    )
    assert store_obj.replace_range(
        xml_text_str=xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=[original_binding_obj],
    )

    with pytest.raises(PerformanceContractError, match="now mapped to pod_b"):
        store_obj.replace_range(
            xml_text_str=xml_text_str,
            query_name_str="ALPHA_DAILY_TWR",
            request_from_date_str="2026-08-07",
            request_to_date_str="2026-08-07",
            binding_obj_list=[remapped_binding_obj],
        )


def test_existing_trusted_start_cannot_disappear_on_sync(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    store_obj = PerformanceStore(db_path_str)
    xml_text_str = _xml_str(
        [("U100", "2026-08-07", 100, 101, 1.0)]
    )
    assert store_obj.replace_range(
        xml_text_str=xml_text_str,
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=[
            PodPerformanceBinding("U100", "pod_a", "2026-08-07", None, True)
        ],
    )

    with pytest.raises(PerformanceContractError, match="trusted return start changed"):
        store_obj.replace_range(
            xml_text_str=xml_text_str,
            query_name_str="ALPHA_DAILY_TWR",
            request_from_date_str="2026-08-07",
            request_to_date_str="2026-08-07",
            binding_obj_list=[
                PodPerformanceBinding("U100", "pod_a", None, None, True)
            ],
        )
    assert len(store_obj.load_daily_row_list()) == 1


def test_historical_range_does_not_require_future_pod_statement() -> None:
    binding_by_account_route_dict = {
        "U100": PodPerformanceBinding(
            "U100", "pod_old", "2026-08-07", None, True
        ),
        "U200": PodPerformanceBinding(
            "U200", "pod_future", "2026-09-01", None, True
        ),
    }
    xml_text_str = _xml_str(
        [("U100", "2026-08-07", 100, 101, 1.0)]
    )

    row_obj_list = parse_flex_change_in_nav_xml(
        xml_text_str,
        binding_by_account_route_dict,
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
    )

    assert [(row_obj.account_route_str, row_obj.market_date_str) for row_obj in row_obj_list] == [
        ("U100", "2026-08-07")
    ]


def test_flow_sensitive_fund_error_keeps_official_pod_rows(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    PerformanceStore(db_path_str).replace_range(
        xml_text_str=_xml_str(
            [
                ("U100", "2026-08-07", 100, 220, 10.0),
                ("U200", "2026-08-07", 100, 100, 0.0),
            ]
        ),
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=list(_binding_dict().values()),
    )

    page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 10, 10, 0, tzinfo=UTC),
    )

    assert page_dict["status_str"] == "error"
    assert page_dict["fund_chart_dict"] is None
    assert len(page_dict["pod_performance_dict_list"]) == 2
    assert "combined history is unavailable" in page_dict["detail_str"]
    assert "Official Pod TWR remains available" in page_dict["detail_str"]


def test_page_dict_keeps_shadow_source_and_pod_breakdown(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    PerformanceStore(db_path_str).replace_range(
        xml_text_str=_xml_str(
            [
                ("U100", "2026-08-07", 100, 101, 1.0),
                ("U200", "2026-08-07", 100, 99, -1.0),
            ]
        ),
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=list(_binding_dict().values()),
        imported_timestamp_str="2026-08-10T10:15:00+00:00",
    )

    page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 10, 10, 0, tzinfo=UTC),
    )

    assert page_dict["status_str"] == "available"
    assert page_dict["query_name_str"] == "ALPHA_DAILY_TWR"
    assert page_dict["coverage_through_date_str"] == "2026-08-07"
    assert page_dict["covered_account_count_int"] == 2
    assert len(page_dict["pod_performance_dict_list"]) == 2
    assert page_dict["fund_chart_dict"] is not None


def test_shadow_freshness_is_pending_before_sla_and_stale_after_sla(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    PerformanceStore(db_path_str).replace_range(
        xml_text_str=_xml_str(
            [
                ("U100", "2026-08-07", 100, 101, 1.0),
                ("U200", "2026-08-07", 100, 99, -1.0),
            ]
        ),
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=list(_binding_dict().values()),
    )

    pending_page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 11, 10, 0, tzinfo=UTC),  # 06:00 ET
    )
    stale_page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),  # 08:00 ET
    )

    assert pending_page_dict["status_str"] == "pending"
    assert pending_page_dict["status_label_str"] == "Performance sync pending"
    assert stale_page_dict["status_str"] == "stale"
    assert "2026-08-10" in stale_page_dict["detail_str"]


def test_missing_database_is_shadow_only_not_initialized(tmp_path) -> None:
    page_dict = build_performance_page_dict(
        str(tmp_path / "missing-performance.sqlite3")
    )

    assert page_dict["status_str"] == "not_initialized"
    assert page_dict["fund_chart_dict"] is None


def test_page_fails_loud_when_history_starts_after_trusted_baseline(tmp_path) -> None:
    db_path_str = str(tmp_path / "performance.sqlite3")
    PerformanceStore(db_path_str).replace_range(
        xml_text_str=_xml_str(
            [
                ("U100", "2026-08-07", 100, 101, 1.0),
                ("U200", "2026-08-07", 100, 99, -1.0),
            ]
        ),
        query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-07",
        binding_obj_list=list(_binding_dict(start_date_str="2026-08-06").values()),
    )

    page_dict = build_performance_page_dict(
        db_path_str,
        as_of_ts=datetime(2026, 8, 10, 10, 0, tzinfo=UTC),
    )

    assert page_dict["status_str"] == "error"
    assert "requested XNYS session 2026-08-06" in page_dict["detail_str"]
