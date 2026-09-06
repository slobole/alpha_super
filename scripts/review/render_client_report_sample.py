"""Render a synthetic PDF for visual QA; never loads client configuration."""

from datetime import UTC, datetime
from pathlib import Path

from alpha.live.client_reporting import build_client_report_dict
from alpha.live.dashboard_v3.demo import build_demo_benchmark_snapshot, build_demo_fixture_tuple
from alpha.live.investor_report import build_investor_snapshot_dict, render_investor_pdf_bytes


def main():
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    client_dict = registry_dict["clients"][1]
    report_dict = build_client_report_dict(
        client_dict, snapshot_dict[client_dict["client_id"]],
        from_date_str="2026-06-01", to_date_str="2026-09-04",
        as_of_ts=datetime(2026, 9, 5, 12, tzinfo=UTC),
        benchmark_snapshot_obj=build_demo_benchmark_snapshot(),
    )
    output_path_obj = Path("output/pdf/live-ops-investor-demo.pdf").resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_bytes(render_investor_pdf_bytes(build_investor_snapshot_dict(report_dict)))
    print(output_path_obj)


if __name__ == "__main__":
    main()
