"""Research-only open-relative DAY-limit experiment; no shared engine mutation.

Signals/selection/quantities use Close_T. The next open fixes L=O*(1-d).
Daily Low is used ONLY by the fill diagnostic, under ideal post-open activation.
LIMIT fill = L; separate cash friction = q*L*0.00025. Native MOO/exit price
friction is unchanged. Stress adds abs(q*fill)*0.001 on every executed side.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int
from strategies.hpi.stateful_long import (
    HPIStatefulLongStrategy, ENTRY_HORIZON_VOTE_STR, TURNOVER_FIELD_STR,
    load_exact_hpi_inputs,
)
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr import (
    DEFAULT_CONFIG, UsSectorEtfIbsDownshockVoxIyrStrategy,
)
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (
    get_us_sector_etf_ibs_downshock_data, resolve_us_sector_etf_execution_calendar_idx,
)
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

STUDY_PATH = REPO_PATH / "results/research/mr_deeper_dip_entry_study"
START_STR = "2010-01-01"
END_STR = "2026-09-14"
DEPTH_TUPLE = (None, 0.005, 0.01, 0.02)
STRATEGY_TUPLE = ("dv2", "hpi235", "sector")
BASE_COST_FLOAT = 0.00025


def sha256_file(source_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk_bytes in iter(lambda: source_file.read(1 << 20), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def write_json(output_path: Path, value_obj: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value_obj, indent=2, ensure_ascii=False,
                                     default=str, allow_nan=False) + "\n", encoding="utf-8")


def policy_name(depth_float: float | None) -> str:
    return "moo" if depth_float is None else f"limit_{depth_float*100:g}pct"


def limit_fill_price(open_float: float, low_float: float, depth_float: float,
                     penetration_float: float = 0.0) -> float | None:
    if not (0.0 < depth_float < 1.0 and 0.0 <= penetration_float < 1.0):
        raise ValueError("Invalid depth or penetration.")
    if not (np.isfinite(open_float) and np.isfinite(low_float)
            and open_float > 0.0 and 0.0 < low_float <= open_float):
        return None
    limit_float = open_float * (1.0 - depth_float)
    # *** CRITICAL*** execution-day Low tests a preselected order only;
    # it must never enter signal, ranking, slots or previous-close sizing.
    return limit_float if low_float <= limit_float*(1.0-penetration_float) else None


class DeeperDipResearchMixin:
    """One isolated adapter; original order generation and accounting stay native."""

    def configure_research(self, depth_float=None, stress_bool=False,
                           signal_df=None) -> None:
        self.depth_float = depth_float
        self.stress_bool = stress_bool
        self.cached_signal_df = signal_df
        self.entry_row_list = []
        self.friction_row_list = []
        self.daily_row_list = []
        self.limit_by_order_dict = {}
        self.entry_by_order_dict = {}
        self.cash_correction_float = 0.0
        self.session_friction_float = 0.0

    def compute_signals(self, pricing_data_df):
        if self.cached_signal_df is not None:
            return self.cached_signal_df
        return super().compute_signals(pricing_data_df)

    def process_orders(self, prices):
        self.limit_by_order_dict = {}
        self.entry_by_order_dict = {}
        self.cash_correction_float = 0.0
        self.session_friction_float = 0.0
        for rank_int, order_obj in enumerate(list(self.get_orders())):
            if not isinstance(order_obj, MarketOrder):
                raise AssertionError("Study expects original market orders.")
            open_float = float(prices.loc[self.current_bar, (order_obj.asset, "Open")])
            sizing_float = self._get_order_sizing_price_float(
                prices, order_obj.asset, open_float)
            quantity_float = order_obj.amount_in_shares(
                sizing_float, float(self.previous_total_value), self.get_position(order_obj.asset))
            if not np.isfinite(quantity_float) or quantity_float <= 0.0:
                continue
            if self.get_position(order_obj.asset) != 0.0:
                raise AssertionError("Unexpected scale-in outside frozen study.")
            low_float = float(prices.loc[self.current_bar, (order_obj.asset, "Low")])
            volume_float = float(prices.loc[self.current_bar, (order_obj.asset, "Volume")])
            if np.isfinite(open_float) and open_float > 0.0 and (
                    not np.isfinite(low_float) or low_float <= 0.0 or low_float > open_float):
                raise AssertionError(
                    f"Unknown limit eligibility, not an expiry: {order_obj.asset} {self.current_bar}")
            row_dict = {
                "date": str(pd.Timestamp(self.current_bar).date()),
                "decision_date": str(pd.Timestamp(self.previous_bar).date()),
                "asset": order_obj.asset, "trade_id": int(order_obj.trade_id),
                "order_id": int(order_obj.id), "selection_order": rank_int,
                "shares": float(quantity_float), "sizing_price": float(sizing_float),
                "prior_nav": float(self.previous_total_value),
                "open": open_float, "low": low_float, "volume": volume_float,
                "status": "submitted", "fill_price": np.nan,
                "native_commission": 0.0, "research_friction": 0.0,
            }
            self.entry_row_list.append(row_dict)
            self.entry_by_order_dict[order_obj.id] = row_dict
            if self.depth_float is not None:
                fill_float = limit_fill_price(
                    open_float, low_float, self.depth_float,
                    0.0005 if self.stress_bool else 0.0)
                if fill_float is None:
                    row_dict["status"] = "expired_or_invalid_bar"
                    self.remove_order(order_obj)
                else:
                    self.limit_by_order_dict[order_obj.id] = fill_float

        super().process_orders(prices)
        # *** CRITICAL*** parent debits its local MARKET transaction value.
        # Correct cash to the same LIMIT value recorded by add_transaction,
        # then debit separately disclosed friction BEFORE any daily metric.
        self.cash += self.cash_correction_float - self.session_friction_float
        self.total_value = self.cash + self.portfolio_value
        for row_dict in self.entry_by_order_dict.values():
            if row_dict["status"] == "submitted":
                row_dict["status"] = "native_canceled"
        positions_ser = self.get_positions()
        self.daily_row_list.append({
            "date": str(pd.Timestamp(self.current_bar).date()),
            "nav": float(self.total_value), "cash": float(self.cash),
            "invested": float(self.portfolio_value),
            "positions": int((positions_ser > 0.0).sum()),
            "research_friction": float(self.session_friction_float),
        })

    def add_transaction(self, trade_id, bar, asset, amount, price, total_value,
                        order_id, commission=0.0):
        corrected_price_float = float(price)
        friction_float = 0.0
        if order_id in self.limit_by_order_dict:
            if amount <= 0.0:
                raise AssertionError("Limit correction must be a positive entry.")
            corrected_price_float = self.limit_by_order_dict[order_id]
            self.cash_correction_float += float(total_value)-float(amount)*corrected_price_float
            friction_float += float(amount)*corrected_price_float*BASE_COST_FLOAT
        corrected_value_float = float(amount)*corrected_price_float
        if self.stress_bool:
            friction_float += abs(corrected_value_float)*0.001
        self.session_friction_float += friction_float
        super().add_transaction(trade_id, bar, asset, amount, corrected_price_float,
                                corrected_value_float, order_id, commission)
        self.friction_row_list.append({
            "date": str(pd.Timestamp(bar).date()), "asset": asset,
            "trade_id": int(trade_id), "order_id": int(order_id),
            "shares": float(amount), "fill_price": corrected_price_float,
            "native_commission": float(commission), "research_friction": friction_float,
        })
        if order_id in self.entry_by_order_dict:
            self.entry_by_order_dict[order_id].update(
                status="filled", fill_price=corrected_price_float,
                native_commission=float(commission), research_friction=friction_float)


class ResearchDV2(DeeperDipResearchMixin, DVO2Strategy):
    pass


class ResearchHPI(DeeperDipResearchMixin, HPIStatefulLongStrategy):
    pass


class ResearchSector(DeeperDipResearchMixin, UsSectorEtfIbsDownshockVoxIyrStrategy):
    pass


def make_strategy(strategy_str: str, universe_df: pd.DataFrame | None, native_bool=False):
    if strategy_str == "dv2":
        strategy_class = DVO2Strategy if native_bool else ResearchDV2
        strategy_obj = strategy_class(
            name="strategy_mr_dv2", benchmarks=["$SPX"], capital_base=100_000.0,
            slippage=BASE_COST_FLOAT, commission_per_share=0.005, commission_minimum=1.0,
            performance_benchmark_adjustment_str="TOTALRETURN")
        strategy_obj.universe_df = universe_df
        strategy_obj.trade_id = 0
        strategy_obj.current_trade = defaultdict(default_trade_id_int)
    elif strategy_str == "hpi235":
        strategy_class = HPIStatefulLongStrategy if native_bool else ResearchHPI
        strategy_obj = strategy_class(
            name="strategy_mr_hpi_sp500_2_3_5_vote", benchmarks=["$SPXTR"],
            ranking_field_str=TURNOVER_FIELD_STR, entry_mode_str=ENTRY_HORIZON_VOTE_STR,
            backtest_start_date_str=START_STR)
        strategy_obj.universe_df = universe_df
    elif strategy_str == "sector":
        config_obj = replace(DEFAULT_CONFIG, backtest_start_date_str=START_STR,
                             end_date_str=END_STR)
        strategy_class = UsSectorEtfIbsDownshockVoxIyrStrategy if native_bool else ResearchSector
        strategy_obj = strategy_class(
            name="strategy_mr_us_sector_etf_ibs_downshock_vox_iyr",
            benchmarks=["$SPX"], config_obj=config_obj)
    else:
        raise ValueError(strategy_str)
    if not native_bool:
        strategy_obj.configure_research()
    return strategy_obj


def prepare_inputs(strategy_str: str) -> None:
    input_path = STUDY_PATH / "data" / strategy_str
    input_path.mkdir(parents=True, exist_ok=True)
    if (input_path / "input_manifest.json").exists():
        raise FileExistsError("Frozen inputs already exist; do not silently overwrite.")
    if strategy_str == "hpi235":
        symbol_list, universe_df, pricing_df = load_exact_hpi_inputs(
            "S&P 500", "$SPXTR", "1998-01-01", END_STR)
    elif strategy_str == "dv2":
        symbol_list, universe_df = build_index_constituent_matrix("S&P 500")
        pricing_df = load_raw_prices(symbol_list, ["$SPX"], "1998-01-01", END_STR)
    else:
        config_obj = replace(DEFAULT_CONFIG, backtest_start_date_str=START_STR,
                             end_date_str=END_STR)
        pricing_df = get_us_sector_etf_ibs_downshock_data(config_obj)
        symbol_list, universe_df = list(config_obj.symbol_tuple), None
    if pricing_df.index.has_duplicates or not pricing_df.index.is_monotonic_increasing:
        raise AssertionError("Price date index invalid.")
    calendar_idx = pd.DatetimeIndex(pricing_df.index[
        (pricing_df.index >= START_STR) & (pricing_df.index <= END_STR)])
    if strategy_str == "sector":
        calendar_idx = calendar_idx.intersection(resolve_us_sector_etf_execution_calendar_idx(
            pricing_df, config_obj))
    calendar_idx = calendar_idx[calendar_idx >= "2010-01-05"]
    if str(calendar_idx[-1].date()) != END_STR:
        raise AssertionError(f"Input end date not frozen end: {calendar_idx[-1]}")
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        str(symbol_str): "TOTALRETURN" if str(symbol_str).startswith("$") else "CAPITALSPECIAL"
        for symbol_str in pricing_df.columns.get_level_values(0).unique()
    }
    pricing_df.to_pickle(input_path / "pricing.pkl")
    if universe_df is not None:
        universe_df.to_pickle(input_path / "universe.pkl")
    pd.Series(calendar_idx).to_csv(input_path / "calendar.csv", index=False, header=["date"])
    metadata_dict = {
        "strategy": strategy_str, "created_at": datetime.now(timezone.utc).isoformat(),
        "vendor": "Norgate local direct", "database_session": END_STR,
        "start": str(calendar_idx[0].date()), "end": str(calendar_idx[-1].date()),
        "session_count": len(calendar_idx), "price_rows": len(pricing_df),
        "symbol_count": len(symbol_list),
        "padding": "NONE_then_Close_valuation_only" if strategy_str == "hpi235" else "native_ALLMARKETDAYS",
        "membership": "exact_PIT" if strategy_str == "hpi235" else (
            "native_PIT_with_last_5_positive_rows_removed_for_noncurrent_members" if strategy_str=="dv2" else "fixed_11_ETFs"),
        "attrs": pricing_df.attrs,
        "sha256": {path_obj.name: sha256_file(path_obj) for path_obj in input_path.iterdir() if path_obj.is_file()},
    }
    write_json(input_path / "input_manifest.json", metadata_dict)
    print(json.dumps({key_str: metadata_dict[key_str] for key_str in
                      ("strategy", "start", "end", "session_count", "symbol_count")}), flush=True)


def load_inputs(strategy_str: str):
    input_path = STUDY_PATH / "data" / strategy_str
    manifest_dict = json.loads((input_path/"input_manifest.json").read_text(encoding="utf-8"))
    for name_str, digest_str in manifest_dict["sha256"].items():
        if sha256_file(input_path/name_str) != digest_str:
            raise AssertionError("Frozen input changed: "+name_str)
    pricing_df = pd.read_pickle(input_path/"pricing.pkl")
    universe_df = pd.read_pickle(input_path/"universe.pkl") if (input_path/"universe.pkl").exists() else None
    calendar_idx = pd.DatetimeIndex(pd.read_csv(input_path/"calendar.csv")["date"])
    return pricing_df, universe_df, calendar_idx


def frame_digest(frame_df: pd.DataFrame) -> str:
    # Order IDs come from a process-global counter, not economic semantics.
    if "order_id" in frame_df.columns:
        frame_df = frame_df.drop(columns=["order_id"])
    return hashlib.sha256(pd.util.hash_pandas_object(frame_df, index=True).values.tobytes()).hexdigest()


def run_cells(strategy_str: str, smoke_bool=False) -> None:
    pricing_df, universe_df, calendar_idx = load_inputs(strategy_str)
    dependency_dict = json.loads((STUDY_PATH/"source_code_manifest.json").read_text(encoding="utf-8"))
    for name_str, digest_str in dependency_dict.items():
        if sha256_file(REPO_PATH/name_str) != digest_str:
            raise AssertionError("Frozen source dependency changed: "+name_str)
    if smoke_bool:
        # Implementation smoke only; no policy selection on this subset.
        calendar_idx = calendar_idx[:35]
    result_path = STUDY_PATH / ("smoke" if smoke_bool else "runs") / strategy_str
    result_path.mkdir(parents=True, exist_ok=True)
    native_path = result_path / "native_parity.json"
    native_obj = make_strategy(strategy_str, universe_df, native_bool=True)
    started_float = time.perf_counter()
    # Native baseline provides the source-correct precomputed signal matrix.
    signal_df = native_obj.compute_signals(pricing_df)
    signal_hash_str = frame_digest(signal_df)
    # Prefix recomputation: use prior historical slices, never future values.
    source_obj = make_strategy(strategy_str, universe_df, native_bool=True)
    audit_rows = []
    tradable_list = [symbol_str for symbol_str in pricing_df.columns.get_level_values(0).unique()
                     if not str(symbol_str).startswith("$")]
    selected_list = tradable_list if strategy_str == "sector" else tradable_list[:12]
    selected_cols = pricing_df.columns.get_level_values(0).isin(selected_list)
    for cutoff_ts in (pd.Timestamp("2015-12-31"), pd.Timestamp("2020-03-31")):
        prefix_df = pricing_df.loc[:cutoff_ts, selected_cols]
        full_subset_df = pricing_df.loc[:, selected_cols]
        prefix_signal_df = source_obj.compute_signals(prefix_df)
        full_subset_signal_df = source_obj.compute_signals(full_subset_df)
        expected_prefix_df = full_subset_signal_df.loc[:cutoff_ts]
        missing_column_idx = expected_prefix_df.columns.difference(prefix_signal_df.columns)
        if expected_prefix_df.loc[:, missing_column_idx].notna().any().any():
            raise AssertionError("Future-created feature column contains past values.")
        # *** CRITICAL*** Schema-only alignment for assets not yet observed:
        # missing prefix features must be ALL NaN, never filled with future data.
        pd.testing.assert_frame_equal(
            prefix_signal_df.reindex(columns=expected_prefix_df.columns),
            expected_prefix_df, check_freq=False, check_dtype=False)
        audit_rows.append({"cutoff": str(cutoff_ts.date()), "symbols": len(selected_list), "pass": True})
    write_json(result_path/"signal_prefix_audit.json", audit_rows)

    # Native compute is still exercised above; reuse identical immutable signals
    # for all economic arms. This assignment is local to one research object.
    native_obj.compute_signals = lambda pricing_data_df: signal_df
    run_daily(native_obj, pricing_df, calendar_idx, show_progress=False,
              show_signal_progress_bool=False)
    native_dict = {
        "transactions_sha256": frame_digest(native_obj.get_transactions()),
        "cash": float(native_obj.cash), "nav": float(native_obj.total_value),
        "source_signal_sha256": signal_hash_str,
        "elapsed_seconds": time.perf_counter()-started_float,
    }
    native_obj.get_transactions().to_csv(result_path/"native_transactions.csv", index=False)
    native_obj.results.to_csv(result_path/"native_equity.csv")
    native_equity_df = native_obj.results.copy()
    del native_obj
    write_json(native_path, native_dict)

    layer_tuple = ("central",) if smoke_bool else ("central", "stress")
    for layer_str in layer_tuple:
        for depth_float in DEPTH_TUPLE:
            cell_str = f"{layer_str}_{policy_name(depth_float)}"
            cell_path = result_path/cell_str
            if (cell_path/"complete.json").exists():
                existing_dict = json.loads((cell_path/"complete.json").read_text(encoding="utf-8"))
                current_hash_dict = {
                    "input_manifest_sha256": sha256_file(STUDY_PATH/"data"/strategy_str/"input_manifest.json"),
                    "code_sha256": sha256_file(Path(__file__)),
                    "spec_sha256": sha256_file(STUDY_PATH/"research_spec_frozen.json"),
                    "dependency_manifest_sha256": sha256_file(STUDY_PATH/"source_code_manifest.json"),
                }
                if any(existing_dict.get(key_str) != value_str for key_str, value_str in current_hash_dict.items()):
                    raise AssertionError("Refusing to mix stale cell with current code/spec/data: "+cell_str)
                print("Existing verified immutable cell:", cell_str, flush=True)
                continue
            cell_path.mkdir(parents=True, exist_ok=True)
            strategy_obj = make_strategy(strategy_str, universe_df)
            strategy_obj.configure_research(depth_float, layer_str=="stress", signal_df)
            started_float = time.perf_counter()
            print("RUN", strategy_str, cell_str, flush=True)
            run_daily(strategy_obj, pricing_df, calendar_idx, show_progress=False,
                      show_signal_progress_bool=False)
            if layer_str == "central" and depth_float is None:
                if frame_digest(strategy_obj.get_transactions()) != native_dict["transactions_sha256"]:
                    raise AssertionError("MOO transaction parity failed.")
                pd.testing.assert_frame_equal(strategy_obj.results, native_equity_df, check_freq=False)
                if not np.isclose(strategy_obj.cash, native_dict["cash"], rtol=0, atol=1e-8):
                    raise AssertionError("MOO cash parity failed.")
            daily_df = pd.DataFrame(strategy_obj.daily_row_list).set_index("date")
            entry_df = pd.DataFrame(strategy_obj.entry_row_list)
            friction_df = pd.DataFrame(strategy_obj.friction_row_list)
            daily_df.to_csv(cell_path/"daily.csv")
            entry_df.to_csv(cell_path/"entries.csv", index=False)
            friction_df.to_csv(cell_path/"friction.csv", index=False)
            strategy_obj.get_transactions().to_csv(cell_path/"transactions.csv", index=False)
            strategy_obj.get_dividend_ledger().to_csv(cell_path/"dividends.csv", index=False)
            strategy_obj.results.to_csv(cell_path/"native_results.csv")
            write_json(cell_path/"complete.json", {
                "strategy": strategy_str, "policy": policy_name(depth_float),
                "depth": depth_float, "layer": layer_str,
                "elapsed_seconds": time.perf_counter()-started_float,
                "nav": float(strategy_obj.total_value),
                "cash": float(strategy_obj.cash),
                "entries": len(entry_df),
                "fills": int(entry_df["status"].eq("filled").sum()) if len(entry_df) else 0,
                "native_moo_parity": depth_float is None and layer_str=="central",
                "input_manifest_sha256": sha256_file(STUDY_PATH/"data"/strategy_str/"input_manifest.json"),
                "code_sha256": sha256_file(Path(__file__)),
                "dependency_manifest_sha256": sha256_file(STUDY_PATH/"source_code_manifest.json"),
                "spec_sha256": sha256_file(STUDY_PATH/"research_spec_frozen.json"),
            })
            print("DONE", strategy_str, cell_str,
                  round(time.perf_counter()-started_float, 2), "seconds", flush=True)
            del strategy_obj
            gc.collect()


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("stage", choices=["prepare", "smoke", "run"])
    parser_obj.add_argument("strategy", choices=STRATEGY_TUPLE)
    args_obj = parser_obj.parse_args()
    if args_obj.stage == "prepare":
        prepare_inputs(args_obj.strategy)
    else:
        run_cells(args_obj.strategy, args_obj.stage == "smoke")


if __name__ == "__main__":
    main()
