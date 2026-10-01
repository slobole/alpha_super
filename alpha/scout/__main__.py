"""Scout command line.

    uv run python -m alpha.scout verify     # walk the ledger hash chain
    uv run python -m alpha.scout summary    # rows per type and trials per family
    uv run python -m alpha.scout health --flex-xml "C:/Users/User/Downloads/ALPHA_DAILY_TWR (1).xml" ...
    uv run python -m alpha.scout health --flex-db C:/alpha/live_ops/ibkr_performance.sqlite3
                                            # live pod health report (report only)
    uv run python -m alpha.scout gate taa_3x [--fresh]   # identity gate: Scout spec vs the real engine
    uv run python -m alpha.scout panel "S&P 500"          # build / refresh a point-in-time panel cache
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from alpha.scout.ledger import DEFAULT_LEDGER_PATH, Ledger, LedgerIntegrityError


def _ledger_command(command_str: str, ledger_path_str: str) -> int:
    ledger = Ledger(ledger_path_str)
    try:
        row_count_int = ledger.verify()
    except LedgerIntegrityError as error:
        print(f"BROKEN: {ledger.ledger_path}: {error}")
        return 1
    if command_str == "verify":
        print(f"OK: {row_count_int} rows, chain intact ({ledger.ledger_path}).")
        return 0

    row_type_counter = Counter(row_dict["row_type_str"] for row_dict in ledger.rows())
    family_trial_counter = Counter(row_dict["family_id_str"] for row_dict in ledger.rows("trial"))
    print(f"Ledger {ledger.ledger_path}: {row_count_int} rows, chain intact.")
    for row_type_str, count_int in sorted(row_type_counter.items()):
        print(f"  {row_type_str:<16} {count_int}")
    if family_trial_counter:
        print("Trials per family:")
        for family_id_str, count_int in family_trial_counter.most_common():
            print(f"  {family_id_str:<36} {count_int}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alpha.scout")
    parser.add_argument("command", choices=("verify", "summary", "health", "gate", "panel"))
    parser.add_argument("spec", nargs="?", default=None, help="gate: taa_3x, ndx_vxn, ndx_atr, ndx_natr20 or ndx_natr20_vxn; panel: index name, e.g. \"S&P 500\"")
    parser.add_argument("--fresh", action="store_true", help="gate: run the engine now instead of the newest saved run")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="ledger path (default: main-checkout ledger)")
    parser.add_argument("--flex-xml", nargs="+", default=None, help="health: IBKR Flex ALPHA_DAILY_TWR XML files")
    parser.add_argument("--flex-db", default=None, help="health: IBKR Flex SQLite store (read-only)")
    parser.add_argument("--out", default=None, help="health: output folder (default results/scout/pod_health/<today>)")
    args = parser.parse_args(argv)

    if args.command in ("verify", "summary"):
        return _ledger_command(args.command, args.ledger)
    if args.command == "panel":
        from alpha.scout.panel import build_panel

        if not args.spec:
            parser.error('panel needs an index name, e.g. "S&P 500" or "Nasdaq 100".')
        print(f"Built {build_panel(args.spec)}")
        return 0
    if args.command == "gate":
        from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate

        if args.spec not in GATED_SPEC_DICT:
            parser.error(f"gate needs one of {sorted(GATED_SPEC_DICT)}.")
        report = run_gate(args.spec, fresh_bool=args.fresh)
        print(report.summary_str())
        return 0 if report.passed_bool else 1

    if bool(args.flex_xml) == bool(args.flex_db):
        parser.error("health needs exactly one of --flex-xml or --flex-db.")
    from alpha.scout.pod_health import run_pod_health_report

    output_dir_path, report_list = run_pod_health_report(
        flex_xml_path_list=[Path(path_str) for path_str in args.flex_xml] if args.flex_xml else None,
        flex_db_path=Path(args.flex_db) if args.flex_db else None,
        output_dir_path=Path(args.out) if args.out else None,
    )
    for report_dict in report_list:
        print(f"{report_dict['label_str']:<10} {report_dict['status_str']}")
    print(f"Report: {output_dir_path / 'pod_health.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
