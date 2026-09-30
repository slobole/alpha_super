"""Scout command line.

    uv run python -m alpha.scout verify     # walk the ledger hash chain
    uv run python -m alpha.scout summary    # rows per type and trials per family
"""

from __future__ import annotations

import argparse
from collections import Counter

from alpha.scout.ledger import DEFAULT_LEDGER_PATH, Ledger, LedgerIntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alpha.scout")
    parser.add_argument("command", choices=("verify", "summary"))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="ledger path (default: repo ledger)")
    args = parser.parse_args(argv)

    ledger = Ledger(args.ledger)
    try:
        row_count_int = ledger.verify()
    except LedgerIntegrityError as error:
        print(f"BROKEN: {ledger.ledger_path}: {error}")
        return 1
    if args.command == "verify":
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


if __name__ == "__main__":
    raise SystemExit(main())
