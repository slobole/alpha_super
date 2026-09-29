"""Frozen-input BIL/SPY account controls for the 2026-09-23 portfolio study.

Research only. Close_T cash buys affordable whole shares at Open_(T+1).
Initial capital is retained as an explicit close anchor, so first-fill costs
and gaps remain in the account return. No vendor, broker or production writes.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import DIVIDEND_LEDGER_COLUMN_TUPLE, Strategy

ROOT_PATH = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT_PATH / "results/research/strategy/strategy_taa_adaptive_macro_core5/snapshot_data_qualification/2026-09-15_step2_final/direct_prices.parquet"
OUTPUT_PATH = ROOT_PATH / "results/research/portfolio_family_20260923/benchmarks"
EXPECTED_INPUT_SHA256_STR = "d355567b94b60f3349f3e6eb42341ac9c7a808de552c494abcc9584256ef4160"
CAPITAL_FLOAT = 1_000_000.0
END_DATE_STR = "2026-09-11"
SLIPPAGE_FLOAT = 0.00025
COMMISSION_PER_SHARE_FLOAT = 0.005
COMMISSION_MINIMUM_FLOAT = 1.0
SOURCE_PATH_TUPLE = (
    "scripts/research/portfolio_family_20260923/benchmarks.py",
    "tests/test_portfolio_family_benchmarks.py",
    "alpha/engine/backtest.py", "alpha/engine/backtester.py",
    "alpha/engine/strategy.py", "alpha/engine/order.py", "alpha/engine/metrics.py",
)


def sha256_str(file_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk_bytes in iter(lambda: input_file.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def write_json(file_path: Path, payload_dict: dict) -> None:
    file_path.write_text(json.dumps(payload_dict, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")


def affordable_shares_int(cash_float: float, close_float: float) -> int:
    """Largest q>=0 with q*Close_T*(1+s)+max(1,.005*q)<=cash_T."""
    if not np.isfinite([cash_float, close_float]).all() or close_float <= 0:
        raise ValueError("Finite cash and positive prior close required.")
    if cash_float <= 0:
        return 0
    unit_budget_float = close_float * (1 + SLIPPAGE_FLOAT)
    share_count_int = max(0, math.floor(cash_float / (unit_budget_float + COMMISSION_PER_SHARE_FLOAT)))
    while share_count_int > 0:
        commission_float = max(COMMISSION_MINIMUM_FLOAT, COMMISSION_PER_SHARE_FLOAT * share_count_int)
        if share_count_int * unit_budget_float + commission_float <= cash_float:
            break
        share_count_int -= 1
    return share_count_int


def common_price_frame_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the entire jointly observed BIL/SPY interval; reject interior holes."""
    if not isinstance(price_df.index, pd.DatetimeIndex) or not price_df.index.is_unique:
        raise ValueError("Unique DatetimeIndex required.")
    if not price_df.index.is_monotonic_increasing:
        raise ValueError("Increasing dates required.")
    adjustment_dict = price_df.attrs.get("norgate_adjustment_by_symbol_dict", {})
    if any(adjustment_dict.get(asset_str) != "CAPITALSPECIAL" for asset_str in ("BIL", "SPY")):
        raise ValueError("BIL/SPY CAPITALSPECIAL provenance required.")
    if adjustment_dict.get("$SPX") != "TOTALRETURN" or price_df.attrs.get("benchmark_data_symbol_dict", {}).get("$SPX") != "$SPXTR":
        raise ValueError("Archived $SPX must explicitly map to TOTALRETURN $SPXTR.")
    column_list = [(asset_str, field_str) for asset_str in ("BIL", "SPY")
                   for field_str in ("Open", "High", "Low", "Close", "Dividend")]
    required_df = price_df.loc[:, column_list].astype(float)
    valid_ser = pd.Series(np.isfinite(required_df.to_numpy()).all(axis=1), index=price_df.index)
    price_column_list = [column_tuple for column_tuple in column_list if column_tuple[1] != "Dividend"]
    valid_ser &= required_df[price_column_list].gt(0).all(axis=1)
    if not valid_ser.any():
        raise ValueError("No jointly observed BIL/SPY interval.")
    first_valid_ts = valid_ser[valid_ser].index[0]
    selected_df = price_df.loc[first_valid_ts:END_DATE_STR].copy()
    if len(selected_df) < 2 or not valid_ser.loc[selected_df.index].all():
        raise ValueError("Missing observed benchmark bar inside common interval.")
    index_price_ser = selected_df[("$SPX", "Close")].astype(float)
    if not np.isfinite(index_price_ser).all() or not index_price_ser.gt(0).all():
        raise ValueError("Missing TOTALRETURN index price.")
    return selected_df


class MonthlyCashReinvestmentStrategy(Strategy):
    """Long-only single-ETF control; residual cash reinvested after month-end."""

    def __init__(self, asset_str: str, capital_float: float = CAPITAL_FLOAT):
        if asset_str not in ("BIL", "SPY") or capital_float <= 0:
            raise ValueError("A positive-capital BIL or SPY control is required.")
        super().__init__(name=f"benchmark_{asset_str.lower()}", benchmarks=["$SPX"],
                         capital_base=capital_float, slippage=SLIPPAGE_FLOAT,
                         commission_per_share=COMMISSION_PER_SHARE_FLOAT,
                         commission_minimum=COMMISSION_MINIMUM_FLOAT,
                         performance_benchmark_symbol_str="$SPX",
                         performance_benchmark_adjustment_str="TOTALRETURN")
        self.asset_str = asset_str
        self.asset_list = [asset_str]
        self.initialized_bool = False
        self.decision_row_list: list[dict] = []
        self.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=0.25)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df.copy()

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series,
                _open_price_ser: pd.Series) -> None:
        # *** CRITICAL *** Calendar months are known before trading. Only cash
        # and prices through Close_T set q; future Open_(T+1) never sizes q.
        month_change_bool = self.previous_bar.to_period("M") != self.current_bar.to_period("M")
        if self.initialized_bool and not month_change_bool:
            return
        close_float = float(close_row_ser[(self.asset_str, "Close")])
        cash_float = float(self.cash)
        share_count_int = affordable_shares_int(cash_float, close_float)
        self.decision_row_list.append({
            "decision_date": self.previous_bar, "fill_date": self.current_bar,
            "asset": self.asset_str, "decision_cash_float": cash_float,
            "decision_close_float": close_float, "buy_shares_int": share_count_int,
            "reason": "initial" if not self.initialized_bool else "monthly_reinvest",
        })
        if share_count_int > 0:
            self.order(self.asset_str, share_count_int, trade_id=1)
        self.initialized_bool = True


def run_control(price_df: pd.DataFrame, asset_str: str,
                capital_float: float = CAPITAL_FLOAT) -> MonthlyCashReinvestmentStrategy:
    strategy_obj = MonthlyCashReinvestmentStrategy(asset_str, capital_float)
    # *** CRITICAL *** Trimmed input starts at the initial capital anchor.
    # Native runner records that close at capital before any next-open order.
    with redirect_stdout(io.StringIO()):
        run_daily(strategy_obj, price_df.copy(), calendar=price_df.index,
                  show_progress=False, show_signal_progress_bool=False,
                  audit_override_bool=False)
    strategy_obj.results.index = pd.DatetimeIndex(strategy_obj.results.index, name="date")
    strategy_obj.realized_weight_df.index = pd.DatetimeIndex(strategy_obj.realized_weight_df.index, name="date")
    return strategy_obj


def validate_account(strategy_obj: MonthlyCashReinvestmentStrategy) -> None:
    nav_df = strategy_obj.results
    if float(nav_df.iloc[0]["total_value"]) != strategy_obj._capital_base or float(nav_df.iloc[0]["portfolio_value"]) != 0:
        raise ValueError("Missing initial capital anchor.")
    if not np.allclose(nav_df["total_value"].astype(float),
                       nav_df["cash"].astype(float) + nav_df["portfolio_value"].astype(float),
                       rtol=1e-11, atol=1e-7):
        raise ValueError("NAV does not reconcile.")
    transaction_df = strategy_obj._transactions
    if transaction_df["amount"].astype(float).le(0).any():
        raise ValueError("Control may only buy positive shares.")
    if len(transaction_df) and pd.Timestamp(transaction_df.iloc[0]["bar"]) <= nav_df.index[0]:
        raise ValueError("First fill must follow capital anchor.")
    ledger_df = pd.DataFrame(strategy_obj._dividend_ledger_row_dict_list, columns=DIVIDEND_LEDGER_COLUMN_TUPLE)
    cash_change_ser = pd.Series(0.0, index=nav_df.index)
    for transaction_row in transaction_df.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(transaction_row.bar)] -= (
            float(transaction_row.amount) * float(transaction_row.price) + float(transaction_row.commission)
        )
    for ledger_row in ledger_df.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(ledger_row.ex_date)] += float(ledger_row.net_dividend_cash_float)
    expected_cash_ser = strategy_obj._capital_base + cash_change_ser.cumsum()
    if not np.allclose(expected_cash_ser, nav_df["cash"].astype(float), rtol=1e-10, atol=1e-6):
        raise ValueError("Cash does not reconcile to native fills, costs and dividends.")
    if not np.allclose(strategy_obj.realized_weight_df.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Native realized weights do not sum to one.")


def export_control(strategy_obj: MonthlyCashReinvestmentStrategy,
                   source_hash_dict: dict, input_hash_str: str) -> dict:
    validate_account(strategy_obj)
    output_path = OUTPUT_PATH / strategy_obj.name
    output_path.mkdir(parents=True, exist_ok=True)
    nav_df = strategy_obj.results.loc[:, ["total_value", "portfolio_value", "cash", "daily_returns"]].copy()
    dividend_df = pd.DataFrame(strategy_obj._dividend_ledger_row_dict_list, columns=DIVIDEND_LEDGER_COLUMN_TUPLE)
    frame_dict = {
        "nav": (nav_df, True), "transactions": (strategy_obj._transactions, False),
        "realized_weights": (strategy_obj.realized_weight_df, True),
        "dividends": (dividend_df, False), "decisions": (pd.DataFrame(strategy_obj.decision_row_list), False),
    }
    output_file_dict = {}
    for label_str, (export_df, index_bool) in frame_dict.items():
        output_file_path = output_path / f"{label_str}.csv.gz"
        export_df.to_csv(output_file_path, index=index_bool, index_label="date" if index_bool else None,
                         compression={"method": "gzip", "mtime": 0})
        output_file_dict[label_str] = {"path_str": str(output_file_path.resolve()), "sha256_str": sha256_str(output_file_path)}
    metadata_dict = {
        "source_id_str": strategy_obj.name,
        "actual_start_date_str": str(nav_df.index[0].date()),
        "actual_end_date_str": str(nav_df.index[-1].date()), "native_row_count_int": len(nav_df),
        "native_capital_float": float(strategy_obj._capital_base),
        "slippage_per_side_float": strategy_obj._slippage,
        "commission_per_share_float": strategy_obj._commission_per_share,
        "commission_minimum_float": strategy_obj._commission_minimum,
        "accounting_policy_dict": dict(strategy_obj._accounting_policy_dict),
        "data_adjustment_policy_dict": dict(strategy_obj._data_adjustment_policy_dict),
        "saved_metadata_source_hash_available_bool": True,
        "current_module_hash_is_historical_run_proof_bool": True,
        "realized_weights_available_bool": True, "cash_weight_column_str": "Cash",
        "weight_null_semantics_str": "native_sparse_unheld_asset_cells",
        "gross_exposure_rule_str": "sum(abs(asset weights)); exclude Cash",
        "benchmark_column_list": [],
        "benchmark_data_symbol_map_dict": {"$SPX": "$SPXTR"},
        "benchmark_basis_warning_str": "Executable ETF account with25% long-dividend withholding; separate index companion is gross.",
        "initial_anchor_note_str": "First row is uninvested initial capital at the earliest jointly valid BIL/SPY close; first fill is next open and is included in returns.",
        "reinvestment_contract_str": "Initial investment, then prior-month-end cash only at next open; known slippage and commissions reserved; no execution-day dividend anticipation.",
        "negative_cash_note_str": "No planned leverage; opening gaps can make actual cash negative. Native accounting reports it; financing not charged.",
        "input_path_str": str(INPUT_PATH), "input_sha256_str": input_hash_str,
        "current_source_hash_dict": source_hash_dict, "output_file_dict": output_file_dict,
    }
    write_json(output_path / "source_metadata.json", metadata_dict)
    return metadata_dict


def main() -> None:
    # Freeze/check the archived input BEFORE loading it or doing new research.
    input_hash_str = sha256_str(INPUT_PATH)
    if input_hash_str != EXPECTED_INPUT_SHA256_STR:
        raise RuntimeError("Frozen archived input hash changed.")
    source_hash_dict = {path_str: sha256_str(ROOT_PATH / path_str) for path_str in SOURCE_PATH_TUPLE}
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    manifest_dict = {
        "status_str": "running", "started_at_utc_str": datetime.now(timezone.utc).isoformat(),
        "input_path_str": str(INPUT_PATH), "input_sha256_str": input_hash_str,
        "current_source_hash_dict": source_hash_dict,
        "controls": ["benchmark_bil", "benchmark_spy"],
        "capital_float": CAPITAL_FLOAT, "research_only_bool": True,
        "rule_str": "Long ETF; first joint close capital anchor; next-open initial buy; monthly prior-close residual-cash reinvest;25% long dividends;2.5bp slippage and0.005/share min1USD.",
    }
    write_json(OUTPUT_PATH / "benchmark_manifest.json", manifest_dict)
    price_df = common_price_frame_df(pd.read_parquet(INPUT_PATH))
    manifest_dict["source_metadata_list"] = [
        export_control(run_control(price_df, asset_str), source_hash_dict, input_hash_str)
        for asset_str in ("BIL", "SPY")
    ]
    index_close_ser = price_df[("$SPX", "Close")].astype(float)
    index_df = pd.DataFrame({
        "gross_totalreturn_index_close": index_close_ser,
        "capital_normalized_gross_index": CAPITAL_FLOAT * index_close_ser / float(index_close_ser.iloc[0]),
    })
    index_df.index.name = "date"
    index_path = OUTPUT_PATH / "spxtr_gross_index.csv.gz"
    index_df.to_csv(index_path, compression={"method": "gzip", "mtime": 0})
    manifest_dict["gross_index_companion_dict"] = {
        "path_str": str(index_path), "sha256_str": sha256_str(index_path),
        "stored_namespace_str": "$SPX", "mapped_vendor_symbol_str": "$SPXTR",
        "adjustment_str": "TOTALRETURN", "basis_str": "Gross total-return index; not an after-tax executable ETF account.",
    }
    if sha256_str(INPUT_PATH) != input_hash_str or any(
        sha256_str(ROOT_PATH / path_str) != hash_str for path_str, hash_str in source_hash_dict.items()
    ):
        raise RuntimeError("Input or current source changed during benchmark computation.")
    manifest_dict["status_str"] = "complete"
    manifest_dict["completed_at_utc_str"] = datetime.now(timezone.utc).isoformat()
    write_json(OUTPUT_PATH / "benchmark_manifest.json", manifest_dict)
    print(json.dumps({"status": "complete", "controls": 2, "rows": len(price_df),
                      "anchor": str(price_df.index[0].date()), "end": str(price_df.index[-1].date())}))


if __name__ == "__main__":
    main()
