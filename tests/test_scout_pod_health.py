"""Live pod health report (alpha/scout/pod_health.py) on synthetic data only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from alpha.scout import pod_health
from alpha.scout.pod_health import (
    ExpectedProcess,
    LivePodSpec,
    completed_month_returns,
    evaluate_pod,
    realized_session_returns,
    render_markdown,
    run_pod_health_report,
    years_to_confirm_edge_loss,
)


@dataclass(frozen=True)
class FakeRow:
    account_route_str: str
    market_date_str: str
    twr_decimal: Decimal


@pytest.fixture
def small_simulation(monkeypatch):
    monkeypatch.setattr(pod_health, "REFERENCE_PATH_COUNT_INT", 6000)
    monkeypatch.setattr(pod_health, "HEALTHY_PATH_COUNT_INT", 800)
    monkeypatch.setattr(pod_health, "DETECTION_PATH_COUNT_INT", 60)


def _expected_process() -> ExpectedProcess:
    date_index = pd.bdate_range("2014-01-02", periods=3000)
    return_ser = pd.Series(np.random.default_rng(0).normal(0.0006, 0.012, 3000), index=date_index)
    return ExpectedProcess(return_ser=return_ser, source_str="synthetic", source_sha256_str="0" * 64)


SPEC = LivePodSpec("pod_test_live_01", "Test pod", "UTEST001", "2026-07-01", lambda root_path, before: _expected_process())
AS_OF = date(2026, 10, 1)


def _live_ser(value_float: float, session_count_int: int) -> pd.Series:
    session_index = pod_health._xnys_sessions("2026-07-01", "2027-09-29")[:session_count_int]
    return pd.Series(value_float, index=session_index)


# ---------------------------------------------------------------- realised returns
def test_non_session_postings_fold_into_the_next_session():
    row_list = [
        FakeRow("UTEST001", "2026-08-28", Decimal("0.01")),  # Friday
        FakeRow("UTEST001", "2026-08-29", Decimal("0.001")),  # Saturday interest posting
        FakeRow("UTEST001", "2026-08-31", Decimal("-0.02")),  # Monday
        FakeRow("UOTHER", "2026-08-31", Decimal("0.5")),
        FakeRow("UTEST001", "2026-08-27", Decimal("0.9")),  # before the monitoring start: ignored
    ]
    realized_ser = realized_session_returns(row_list, "UTEST001", "2026-08-28")
    assert list(realized_ser.index.strftime("%Y-%m-%d")) == ["2026-08-28", "2026-08-31"]
    assert realized_ser.iloc[0] == pytest.approx(0.01)  # the pre-start row is not folded into the first session
    assert realized_ser.iloc[1] == pytest.approx(1.001 * 0.98 - 1)


def test_trailing_non_session_row_is_not_booked_on_a_future_session():
    row_list = [FakeRow("UTEST001", "2026-08-28", Decimal("0.01")), FakeRow("UTEST001", "2026-08-29", Decimal("0.001"))]
    realized_ser = realized_session_returns(row_list, "UTEST001", "2026-08-28")
    assert list(realized_ser.index.strftime("%Y-%m-%d")) == ["2026-08-28"]


def test_conflicting_duplicate_rows_raise():
    row_list = [FakeRow("UTEST001", "2026-08-28", Decimal("0.01")), FakeRow("UTEST001", "2026-08-28", Decimal("0.02"))]
    with pytest.raises(ValueError, match="Conflicting"):
        realized_session_returns(row_list, "UTEST001", "2026-08-28")
    same_row_list = [FakeRow("UTEST001", "2026-08-28", Decimal("0.01"))] * 2
    assert realized_session_returns(same_row_list, "UTEST001", "2026-08-28").size == 1


def test_missing_sessions_raise_instead_of_hiding_returns():
    session_list = pod_health._xnys_sessions("2026-08-03", "2026-08-31")
    row_list = [FakeRow("UTEST001", day.strftime("%Y-%m-%d"), Decimal("0.001")) for day in session_list if day.day not in (12, 13)]
    with pytest.raises(ValueError, match="2 session"):
        realized_session_returns(row_list, "UTEST001", "2026-08-03")
    # A monitoring start before the first row also counts as a gap.
    with pytest.raises(ValueError, match="missing"):
        realized_session_returns(row_list[5:], "UTEST001", "2026-08-03")


def test_completed_month_returns_need_the_first_and_last_session():
    daily_ser = pd.Series(0.001, index=pod_health._xnys_sessions("2026-07-01", "2026-09-15"))
    assert [str(period) for period in completed_month_returns(daily_ser).index] == ["2026-07", "2026-08"]
    through_august_ser = daily_ser.loc[:"2026-08-31"]
    assert [str(period) for period in completed_month_returns(through_august_ser).index] == ["2026-07", "2026-08"]
    mid_july_start_ser = daily_ser.loc["2026-07-15":]
    assert [str(period) for period in completed_month_returns(mid_july_start_ser).index] == ["2026-08"]


# ---------------------------------------------------------------- statistics
def test_years_to_confirm_edge_loss_formula():
    return_vec = np.random.default_rng(1).normal(0.0005, 0.01, 50_000)
    mean_float, std_float = return_vec.mean(), return_vec.std(ddof=1)
    expected_years_float = ((1.6448536 + 0.8416212) * std_float / mean_float) ** 2 / 252
    assert years_to_confirm_edge_loss(return_vec) == pytest.approx(expected_years_float, rel=1e-6)
    assert years_to_confirm_edge_loss(-np.abs(return_vec)) == float("inf")
    # Mean-reverting days (negative autocorrelation) need fewer years once the block variance is used.
    shock_vec = np.random.default_rng(2).normal(0, 0.01, 50_001)
    reverting_vec = 0.0005 + shock_vec[1:] - 0.5 * shock_vec[:-1]
    assert years_to_confirm_edge_loss(reverting_vec, block_length_int=63) < years_to_confirm_edge_loss(reverting_vec)


# ---------------------------------------------------------------- evaluation
def test_statuses(small_simulation):
    expected = _expected_process()
    healthy_report = evaluate_pod(SPEC, expected, _live_ser(0.0008, 60), AS_OF)
    assert healthy_report["status_str"] == "GREEN"
    assert healthy_report["live"]["first_red_date_str"] is None
    assert 0.0 < healthy_report["healthy_false_alarm"]["either_share_float"] < 0.2

    crash_report = evaluate_pod(SPEC, expected, _live_ser(-0.006, 60), AS_OF)
    assert crash_report["status_str"] == "RED"
    assert crash_report["live"]["current_drawdown_age_int"] == 60
    assert crash_report["live"]["first_red_date_str"] is not None

    early_report = evaluate_pod(SPEC, expected, _live_ser(-0.01, 10), date(2026, 7, 16))
    assert early_report["status_str"] == "TOO_EARLY"
    empty_report = evaluate_pod(SPEC, expected, pd.Series(dtype=float), AS_OF)
    assert empty_report["status_str"] == "NO_DATA"

    stale_report = evaluate_pod(SPEC, expected, _live_ser(0.0008, 30), AS_OF)
    assert stale_report["status_str"] == "STALE" and stale_report["live"]["stale_bool"]

    # A crash that recovered: not RED today, but the earlier RED stays on record.
    recovered_ser = pd.Series(np.r_[np.full(60, -0.006), np.full(60, 0.02)], index=_live_ser(0.0, 120).index)
    recovered_report = evaluate_pod(SPEC, expected, recovered_ser, date(2026, 12, 20))
    assert recovered_report["status_str"] != "RED" or recovered_report["live"]["cusum_alarm_bool"]
    assert recovered_report["live"]["first_red_date_str"] is not None

    markdown_str = render_markdown([healthy_report, crash_report, early_report, empty_report, stale_report], "2026-10-01", "synthetic")
    assert "**RED**" in markdown_str and "**GREEN**" in markdown_str and "**STALE**" in markdown_str
    assert "sessions behind" in markdown_str
    assert "Years of data needed to confirm the edge is gone" in markdown_str


def test_expected_process_ends_before_the_monitoring_start(small_simulation):
    expected = _expected_process()
    spec = LivePodSpec("pod_x", "X", "UTEST001", str(expected.return_ser.index[2500].date()), SPEC.expected_loader)
    report_dict = evaluate_pod(spec, expected, pd.Series(0.0005, index=pd.bdate_range(expected.return_ser.index[2500], periods=30)), AS_OF)
    assert report_dict["expected"]["end_str"] < spec.monitoring_start_date_str
    assert report_dict["expected"]["session_count_int"] == 2500


def _flex_xml_str(account_str: str, day_twr_list: list[tuple[str, float]]) -> str:
    statement_list = [
        f'<FlexStatement accountId="{account_str}" fromDate="{day}" toDate="{day}">'
        f'<AccountInformation accountId="{account_str}" currency="USD" />'
        f'<ChangeInNAV accountId="{account_str}" currency="USD" fromDate="{day}" toDate="{day}" '
        f'startingValue="1000" endingValue="{1000 * (1 + twr / 100):.4f}" twr="{twr}" /></FlexStatement>'
        for day, twr in day_twr_list
    ]
    return (
        '<FlexQueryResponse queryName="ALPHA_DAILY_TWR" type="AF"><FlexStatements>'
        + "".join(statement_list)
        + "</FlexStatements></FlexQueryResponse>"
    )


def test_run_report_from_flex_xml(tmp_path, small_simulation):
    session_list = [day.strftime("%Y%m%d") for day in pod_health._xnys_sessions("2026-07-01", "2026-08-31")]
    xml_path = tmp_path / "flex.xml"
    xml_path.write_text(_flex_xml_str("UTEST001", [(day, 0.05) for day in session_list]), encoding="utf-8")
    output_dir_path, report_list = run_pod_health_report(
        flex_xml_path_list=[xml_path], output_dir_path=tmp_path / "out", pod_spec_tuple=(SPEC,), as_of_date=date(2026, 9, 2)
    )
    assert report_list[0]["live"]["session_count_int"] == len(session_list)
    assert report_list[0]["status_str"] == "GREEN"
    assert (output_dir_path / "pod_health.md").exists() and (output_dir_path / "pod_health.json").exists()
    with pytest.raises(ValueError, match="either"):
        run_pod_health_report(pod_spec_tuple=(SPEC,))



def test_status_precedence_and_stale_boundary(small_simulation, monkeypatch):
    expected = _expected_process()
    # RED beats STALE.
    assert evaluate_pod(SPEC, expected, _live_ser(-0.006, 60), date(2027, 3, 1))["status_str"] == "RED"
    # Stale boundary: data end 2026-08-12 (30 sessions); 5 sessions behind is fresh, 6 is stale.
    live_ser = _live_ser(0.0008, 30)
    session_index = pod_health._xnys_sessions("2026-08-13", "2026-09-30")
    assert evaluate_pod(SPEC, expected, live_ser, session_index[5].date())["status_str"] == "GREEN"
    assert evaluate_pod(SPEC, expected, live_ser, session_index[6].date())["status_str"] == "STALE"
    # A CUSUM alarm shows even before the CBI starts (TOO_EARLY would otherwise hide it).
    monkeypatch.setattr(pod_health, "MIN_OBSERVATION_INT", 200)
    cusum_only_report = evaluate_pod(SPEC, expected, _live_ser(-0.004, 64), date(2026, 10, 1))
    assert cusum_only_report["live"]["current_cbi_float"] is None
    assert cusum_only_report["live"]["cusum_alarm_bool"]
    assert cusum_only_report["status_str"] == "RED"


def test_reported_live_numbers_have_known_answers(small_simulation):
    expected = _expected_process()
    return_vec = np.r_[np.full(40, 0.001), np.full(30, -0.008), np.full(50, 0.012)]
    live_ser = pd.Series(return_vec, index=_live_ser(0.0, 120).index)
    report_dict = evaluate_pod(SPEC, expected, live_ser, date(2026, 12, 18))
    live_dict = report_dict["live"]
    assert live_dict["cumulative_return_float"] == pytest.approx(np.prod(1 + return_vec) - 1)
    assert live_dict["current_drawdown_float"] == pytest.approx(0.0) and live_dict["current_drawdown_age_int"] == 0
    # The worst CBI is at the bottom of the crash (session 70), the current one is back to 1.
    assert live_dict["worst_cbi_date_str"] == str(live_ser.index[69].date())
    assert live_dict["current_cbi_float"] == 1.0 and live_dict["worst_cbi_float"] < live_dict["current_cbi_float"]
    first_red_str = live_dict["first_red_date_str"]
    assert first_red_str is not None and live_ser.index[40] <= pd.Timestamp(first_red_str) <= live_ser.index[69]


def test_detector_rates_combine_with_or_and_count_sessions_from_one(small_simulation, monkeypatch):
    from alpha.stats.pod_monitor import build_cbi_table, calibrate_cbi_thresholds, cbi_path
    from alpha.stats.health import calibrate_cusum_threshold

    expected_vec = _expected_process().return_ser.to_numpy()
    table = build_cbi_table(expected_vec, 252, path_count_int=6000)
    thresholds = calibrate_cbi_thresholds(expected_vec, table, healthy_path_count_int=800)
    cusum_calibration = calibrate_cusum_threshold(np.random.default_rng(0).normal(0.01, 0.05, 200))
    crash_vec, calm_vec = np.full(252, -0.006), np.full(252, 0.001)
    monkeypatch.setattr(pod_health, "_cusum_alarm_vec", lambda path_mat, calibration: np.array([False, True]))
    rate_dict = pod_health._detector_rates(np.vstack([crash_vec, calm_vec]), table, thresholds, cusum_calibration)
    assert rate_dict["cbi_red_share_float"] == 0.5
    assert rate_dict["cusum_alarm_share_float"] == 0.5
    assert rate_dict["either_share_float"] == 1.0
    first_red_session_int = int(np.flatnonzero(cbi_path(crash_vec, table) < thresholds.red_float)[0]) + 1
    assert rate_dict["median_sessions_to_cbi_red_among_detected_float"] == first_red_session_int


def test_scenario_drift_shifts(small_simulation, monkeypatch):
    captured_shift_list = []
    original_fn = pod_health.simulated_live_path_mat

    def capture_fn(expected_vec, path_count_int, block_float, length_int, seed_int, daily_drift_shift_float=0.0):
        captured_shift_list.append(daily_drift_shift_float)
        return original_fn(expected_vec, path_count_int, block_float, length_int, seed_int, daily_drift_shift_float)

    monkeypatch.setattr(pod_health, "simulated_live_path_mat", capture_fn)
    expected = _expected_process()
    evaluate_pod(SPEC, expected, pd.Series(dtype=float), AS_OF)
    mean_float = expected.return_ser.loc[: "2026-06-30"].mean()
    assert captured_shift_list == pytest.approx([0.0, -mean_float, -2 * mean_float, -4 * mean_float])


def test_long_live_history_extends_the_table(small_simulation):
    report_dict = evaluate_pod(SPEC, _expected_process(), _live_ser(0.0008, 300), date(2027, 9, 10))
    assert report_dict["live"]["session_count_int"] == 300
    assert report_dict["status_str"] in ("GREEN", "AMBER")


def test_ndx_loader_picks_the_worst_pre_live_offset_and_trims_warm_up(tmp_path):
    date_index = pd.bdate_range("2020-01-01", periods=400)
    rng_obj = np.random.default_rng(0)
    column_dict = {}
    for k_int in range(-10, 11):
        return_vec = rng_obj.normal(0.001, 0.01, 400)
        return_vec[:5] = 0.0
        column_dict[f"ROC12/ATR20_LIVE|N10EW|F100|GSPY|b0|k{k_int:+d}|V22-0.25"] = return_vec
    offset_df = pd.DataFrame(column_dict, index=date_index)
    before_date = date_index[300]
    # k-3 is the worst before the cut but spectacular after it.
    offset_df.iloc[5:300, offset_df.columns.get_loc("ROC12/ATR20_LIVE|N10EW|F100|GSPY|b0|k-3|V22-0.25")] -= 0.005
    offset_df.iloc[300:, offset_df.columns.get_loc("ROC12/ATR20_LIVE|N10EW|F100|GSPY|b0|k-3|V22-0.25")] += 0.05
    parquet_path = tmp_path / "results/research/ndx_param_robustness_20260926/returns_NDX_engine.parquet"
    parquet_path.parent.mkdir(parents=True)
    offset_df.to_parquet(parquet_path)
    expected = pod_health.load_ndx_vxn_expected(tmp_path, before_date)
    assert "(k-3)" in expected.source_str
    assert expected.return_ser.index[0] == date_index[5] and expected.return_ser.index[-1] < before_date
    assert len(expected.source_sha256_str) == 64


def test_taa_loader_uses_the_newest_pickle(tmp_path, monkeypatch):
    from alpha.engine import strategy as strategy_module

    for stamp_str in ("2026-05-20_144258", "2026-09-30_102040", "2026-07-01_100614"):
        folder_path = tmp_path / "results/research/strategy/strategy_taa_df_btal_fallback_tqqq_vix_cash/vanilla_backtest" / stamp_str
        folder_path.mkdir(parents=True)
        (folder_path / "strategy_taa_df_btal_fallback_tqqq_vix_cash.pkl").write_bytes(stamp_str.encode())
    read_path_list = []

    class FakeStrategy:
        def __init__(self, path_str):
            read_path_list.append(path_str)
            self.results = pd.DataFrame({"total_value": [100.0, 101.0, 102.0, 103.0]}, index=pd.bdate_range("2026-06-25", periods=4))

    monkeypatch.setattr(strategy_module.Strategy, "read_pickle", staticmethod(lambda path_str: FakeStrategy(path_str)))
    expected = pod_health.load_taa_expected(tmp_path, pd.Timestamp("2026-07-01"))
    assert "2026-09-30_102040" in read_path_list[0]
    assert expected.return_ser.index[-1] < pd.Timestamp("2026-07-01") and expected.return_ser.size == 3


def test_health_cli_needs_exactly_one_source(capsys):
    from alpha.scout.__main__ import main as scout_cli_main

    with pytest.raises(SystemExit):
        scout_cli_main(["health"])
    with pytest.raises(SystemExit):
        scout_cli_main(["health", "--flex-xml", "a.xml", "--flex-db", "b.sqlite3"])
