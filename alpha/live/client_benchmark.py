"""Saved total-return benchmark measurement; no fetching or execution inputs.

The selected snapshot is explicit and its exact bytes are hashed before parsing.
This is retrospective market measurement, never decision-time replay evidence.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re

from alpha.live.scheduler_utils import get_exchange_calendar_obj


BENCHMARK_LABEL_DICT = {"SPY": "SPY total return", "$SPXTR": "S&P 500 total return"}
BENCHMARK_METHOD_STR = "saved_total_return_close_v1"


@dataclass(frozen=True)
class BenchmarkSnapshot:
    symbol_str: str = ""
    close_tuple: tuple[tuple[str, float | None], ...] = ()
    manifest_hash_str: str = ""
    price_hash_str: str = ""
    snapshot_date_str: str = ""
    profile_str: str = ""
    is_demo_bool: bool = False
    unavailable_reason_str: str | None = None


def validate_benchmark_config(benchmark_dict):
    if benchmark_dict is None:
        return
    if not isinstance(benchmark_dict, dict) or set(benchmark_dict) != {"symbol", "snapshot_directory"}:
        raise ValueError("Benchmark requires symbol and a server-configured snapshot_directory.")
    if not isinstance(benchmark_dict["symbol"], str) or benchmark_dict["symbol"] not in BENCHMARK_LABEL_DICT or not isinstance(benchmark_dict["snapshot_directory"], str) or not benchmark_dict["snapshot_directory"].strip():
        raise ValueError("Benchmark must select SPY or $SPXTR and an explicit saved snapshot directory.")


def load_benchmark_snapshot(benchmark_dict):
    """Read a pinned Norgate manifest/parquet pair; never use a 'latest' lookup.

    File bytes (not just their names/mtime) identify the result. A concurrent
    replacement either matches the pinned manifest or fails closed. No process
    environment, Norgate service, cache, SQLite or production file is changed.
    """
    validate_benchmark_config(benchmark_dict)
    if benchmark_dict is None:
        return BenchmarkSnapshot(unavailable_reason_str="No saved market benchmark is configured.")
    symbol_str = benchmark_dict["symbol"]
    try:
        directory_obj = Path(benchmark_dict["snapshot_directory"])
        with (directory_obj / "manifest.json").open("rb") as manifest_file:
            manifest_bytes = manifest_file.read(1_000_001)
        if len(manifest_bytes) > 1_000_000:
            raise ValueError("Oversize manifest")
        manifest_dict = json.loads(manifest_bytes)
        if type(manifest_dict.get("schema_version")) is not int or manifest_dict["schema_version"] not in {1, 2}:
            raise ValueError("Unsupported snapshot schema")
        snapshot_date_str = manifest_dict["snapshot_market_session_date_str"]
        if snapshot_date_str != date.fromisoformat(snapshot_date_str).isoformat() or snapshot_date_str != directory_obj.name or not isinstance(manifest_dict["profile"], str) or not re.fullmatch(r"[a-z0-9_]{1,128}", manifest_dict["profile"]):
            raise ValueError("Snapshot identity mismatch")
        expected_hash_str = manifest_dict["files"]["prices.parquet"]["sha256"]
        # Bound memory use when reading an operator-selected artifact. Parse the
        # same immutable byte buffer that was checked, not a later filesystem read.
        with (directory_obj / "prices.parquet").open("rb") as price_file:
            price_bytes = price_file.read(512_000_001)
        if len(price_bytes) > 512_000_000 or sha256(price_bytes).hexdigest() != expected_hash_str:
            raise ValueError("Price size/hash mismatch")
        import pandas as pd

        price_df = pd.read_parquet(BytesIO(price_bytes), columns=["date", "symbol_str", "adjustment_str", "Close"],
            filters=[("symbol_str", "==", symbol_str), ("adjustment_str", "==", "TOTALRETURN")])
        close_list, date_set = [], set()
        for row_obj in price_df.itertuples(index=False):
            if row_obj.symbol_str != symbol_str or row_obj.adjustment_str != "TOTALRETURN":
                raise ValueError("Unexpected benchmark adjustment")
            timestamp_obj = pd.Timestamp(row_obj.date)
            if pd.isna(timestamp_obj) or timestamp_obj.tzinfo is not None or timestamp_obj != timestamp_obj.normalize():
                raise ValueError("Benchmark dates must be unambiguous daily session labels")
            date_str = timestamp_obj.date().isoformat()
            if date_str in date_set or date_str > snapshot_date_str:
                raise ValueError("Duplicate/future benchmark date")
            date_set.add(date_str)
            close_float = float(row_obj.Close)
            close_list.append((date_str, close_float if math.isfinite(close_float) and close_float > 0 else None))
        if not close_list:
            raise ValueError("Missing total-return benchmark")
        return BenchmarkSnapshot(symbol_str, tuple(sorted(close_list)), sha256(manifest_bytes).hexdigest(), expected_hash_str,
            snapshot_date_str, manifest_dict["profile"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return BenchmarkSnapshot(symbol_str=symbol_str, unavailable_reason_str="Configured benchmark snapshot could not be validated. No substitute source was used.")


def account_benchmark_dict(strategy_dict, snapshot_obj, *, as_of_date_str=None):
    """Benchmark R = TR_close_end / TR_close_before_first_return - 1.

    Compare against the full official account TWR, including non-session rows.
    A closed-market endpoint explicitly uses the last exchange-session close;
    missing open-session prices are never filled or removed from the interval.
    Difference_pp = 100 * (account_R - benchmark_R), not dollars or model alpha.
    """
    snapshot_obj = snapshot_obj or BenchmarkSnapshot(unavailable_reason_str="No saved market benchmark is configured.")
    result_dict = {
        "status_str": "unavailable", "label_str": BENCHMARK_LABEL_DICT.get(snapshot_obj.symbol_str, "Market benchmark"),
        "method_str": BENCHMARK_METHOD_STR, "symbol_str": snapshot_obj.symbol_str,
        "adjustment_str": "TOTALRETURN", "currency_str": "USD", "is_demo_bool": snapshot_obj.is_demo_bool,
        "from_date_str": strategy_dict["from_date_str"], "to_date_str": strategy_dict["to_date_str"],
        "baseline_date_str": None, "end_price_date_str": None, "return_float": None, "difference_pp_float": None,
        "manifest_hash_str": snapshot_obj.manifest_hash_str, "price_hash_str": snapshot_obj.price_hash_str,
        "snapshot_date_str": snapshot_obj.snapshot_date_str, "profile_str": snapshot_obj.profile_str,
        "reason_str": snapshot_obj.unavailable_reason_str,
        "basis_str": "Full account interval versus a USD total-return close benchmark. Closed-market endpoints use the last exchange close; no missing session price is filled. No additional account trading/advisory fees or investor-specific tax are deducted from the benchmark. Not dollar P&L, risk-adjusted alpha or execution replay.",
    }
    if snapshot_obj.unavailable_reason_str:
        return result_dict
    if as_of_date_str is not None and snapshot_obj.snapshot_date_str > as_of_date_str:
        result_dict["reason_str"] = "Benchmark snapshot is future-dated relative to the reporting assessment."
        return result_dict
    if not strategy_dict["coverage_complete_bool"] or strategy_dict["twr_float"] is None:
        result_dict["reason_str"] = "Full-period official account returns are unavailable; no shortened intersection is used."
        return result_dict
    try:
        calendar_obj = get_exchange_calendar_obj("XNYS")
        prior_day_str = (date.fromisoformat(strategy_dict["from_date_str"]) - timedelta(days=1)).isoformat()
        baseline_obj = calendar_obj.date_to_session(prior_day_str, direction="previous")
        end_obj = calendar_obj.date_to_session(strategy_dict["to_date_str"], direction="previous")
        required_date_list = [session_obj.date().isoformat() for session_obj in calendar_obj.sessions_in_range(baseline_obj, end_obj)]
        close_dict = dict(snapshot_obj.close_tuple)
        if len(close_dict) != len(snapshot_obj.close_tuple) or not required_date_list or any(close_dict.get(date_str) is None or not math.isfinite(close_dict[date_str]) or close_dict[date_str] <= 0 for date_str in required_date_list):
            raise ValueError("Incomplete benchmark coverage")
        # *** CRITICAL*** Retrospective performance boundary, not a signal join:
        # include the previous session's close so the first account day is not
        # lost. Keep weekend/holiday account returns in the official account TWR.
        return_decimal = Decimal(str(close_dict[required_date_list[-1]])) / Decimal(str(close_dict[required_date_list[0]])) - 1
        return_float = float(return_decimal)
        difference_pp_float = float((Decimal(str(strategy_dict["twr_float"])) - return_decimal) * 100)
        if not math.isfinite(return_float) or not math.isfinite(difference_pp_float):
            raise ValueError("Non-finite benchmark result")
        result_dict.update(status_str="ready", baseline_date_str=required_date_list[0], end_price_date_str=required_date_list[-1],
            return_float=return_float, difference_pp_float=difference_pp_float, reason_str=None)
    except (ValueError, KeyError, TypeError, OverflowError):
        result_dict["reason_str"] = "Benchmark coverage or numeric values cannot support the exact account interval. Comparison is withheld."
    return result_dict
