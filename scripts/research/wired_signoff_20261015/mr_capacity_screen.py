"""Screen cached MR fills versus causal native Turnover at constant USD1M NAV.

This is an ADV participation diagnostic, not an opening-auction impact model.
Synthetic last-close liquidations remain separate from auction fills.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH_OBJ))


def _stats(frame_df):
    valid_ser = frame_df["participation_float"].dropna()
    return {"fill_count_int": len(frame_df), "valid_participation_count_int": len(valid_ser),
        "median_order_adv_fraction_float": None if valid_ser.empty else float(valid_ser.median()),
        "p99_order_adv_fraction_float": None if valid_ser.empty else float(valid_ser.quantile(.99)),
        "max_order_adv_fraction_float": None if valid_ser.empty else float(valid_ser.max()),
        "fraction_above_one_percent_adv_float": None if valid_ser.empty else float(valid_ser.gt(.01).mean()),
        "fraction_above_house_moo_point_one_percent_adv_float": None if valid_ser.empty else float(valid_ser.gt(.001).mean())}


def main():
    audit_root_path_obj = REPO_PATH_OBJ / "results/review/wired_signoff_20261015"
    report_dict = {"generated_at_utc_str": datetime.now(timezone.utc).isoformat(), "capital_float": 1_000_000,
        "checker_sha256_str": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formula_str": "order_notional_at_constant_1M = abs(filled_ledger_shares * fill_price) / engine_NAV_Close_T * 1M; participation=that_notional/median(native_Turnover_last20sessions_throughT)",
        "timing_str": "T is recorded decision_date, and fill session is recorded execution_date; no execution-day Turnover enters its ADV denominator.",
        "limitations_list": ["ADV participation alone does not prove auction liquidity, available quotes, borrow, fills or market impact.",
            "Constant1M-NAV screen differs from a1M-initial-capital account that compounds for decades.",
            "Only completed actual engine fill records are screened; synthetic last-close liquidations are counted separately."],
        "strategy_list": []}
    for baseline_path_obj in sorted((audit_root_path_obj / "baselines").glob("strategy_mr*")):
        trace_input_path_obj = baseline_path_obj
        if baseline_path_obj.name == "strategy_mr_hpi_sp500_2_3_5_vote" and not (trace_input_path_obj / "trace.pkl.gz").exists():
            trace_input_path_obj = audit_root_path_obj / "mr_hpi235_trace_only"
            qualification_dict = json.loads((trace_input_path_obj / "qualification.json").read_text())
            assert qualification_dict["status_str"] == "pass" and qualification_dict["exact_original_baseline_equity_bool"]
        required_name_tuple = ("transactions.csv.gz", "trace.pkl.gz", "pricing.parquet")
        if not (baseline_path_obj / required_name_tuple[0]).exists() or not all((trace_input_path_obj / name_str).exists() for name_str in required_name_tuple[1:]):
            continue
        pricing_df = pd.read_parquet(trace_input_path_obj / "pricing.parquet")
        transaction_df = pd.read_csv(baseline_path_obj / "transactions.csv.gz")
        trace_obj = pd.read_pickle(trace_input_path_obj / "trace.pkl.gz")
        trace_list = [{**record_dict, "decision_date": decision_ts, "nav": record_dict["nav_float"]}
                      for decision_ts, record_dict in trace_obj.items()] if isinstance(trace_obj, dict) else trace_obj
        trace_dict = {pd.Timestamp(row_dict["execution_date"]): row_dict for row_dict in trace_list}
        transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
        synthetic_count_int = int(transaction_df["order_id"].lt(0).sum()) if "order_id" in transaction_df else 0
        if "order_id" in transaction_df:
            transaction_df = transaction_df.loc[transaction_df["order_id"].ge(0)]
        evidence_row_list = []
        for transaction_row_obj in transaction_df.itertuples(index=False):
            execution_ts = pd.Timestamp(transaction_row_obj.bar)
            trace_row_dict = trace_dict.get(execution_ts)
            if trace_row_dict is None:
                evidence_row_list.append({"execution_date_str": execution_ts.date().isoformat(), "asset_str": transaction_row_obj.asset,
                    "participation_float": np.nan, "error_str": "No recorded decision NAV/date for execution."})
                continue
            decision_ts = pd.Timestamp(trace_row_dict["decision_date"])
            turnover_key_tuple = (str(transaction_row_obj.asset), "Turnover")
            nav_float = float(trace_row_dict["nav"])
            # *** CRITICAL *** Liquidity known at Close_T only. The engine may
            # contain future vendor rows; loc[:T] is applied before the20rowtail.
            turnover_ser = (pd.to_numeric(pricing_df.loc[:decision_ts, turnover_key_tuple], errors="coerce").iloc[-20:]
                            if turnover_key_tuple in pricing_df.columns else pd.Series(dtype=float))
            adv_float = float(turnover_ser.median()) if len(turnover_ser) == 20 and np.isfinite(turnover_ser.to_numpy()).all() else np.nan
            notional_float = abs(float(transaction_row_obj.amount) * float(transaction_row_obj.price)) / nav_float * 1_000_000
            participation_float = notional_float / adv_float if np.isfinite(adv_float) and adv_float > 0 else np.nan
            evidence_row_list.append({"execution_date_str": execution_ts.date().isoformat(), "decision_date_str": decision_ts.date().isoformat(),
                "asset_str": str(transaction_row_obj.asset), "constant_1m_order_notional_float": notional_float,
                "causal_adv20_median_dollar_float": adv_float, "participation_float": participation_float})
        evidence_df = pd.DataFrame(evidence_row_list)
        if evidence_df.empty:
            continue
        latest_execution_ts = pd.to_datetime(evidence_df["execution_date_str"]).max()
        recent_df = evidence_df.loc[pd.to_datetime(evidence_df["execution_date_str"]) >= latest_execution_ts-pd.DateOffset(years=3)]
        latest_year_df = evidence_df.loc[pd.to_datetime(evidence_df["execution_date_str"]) >= latest_execution_ts-pd.DateOffset(years=1)]
        report_dict["strategy_list"].append({"module_str": baseline_path_obj.name, "synthetic_last_close_liquidation_count_int": synthetic_count_int,
            "trace_input_path_str": str(trace_input_path_obj.relative_to(REPO_PATH_OBJ)),
            "latest_actual_execution_date_str": latest_execution_ts.date().isoformat(),
            "recent_three_year_window_start_str": (latest_execution_ts-pd.DateOffset(years=3)).date().isoformat(),
            "latest_year_window_start_str": (latest_execution_ts-pd.DateOffset(years=1)).date().isoformat(),
            "transactions_sha256_str": hashlib.sha256((baseline_path_obj / "transactions.csv.gz").read_bytes()).hexdigest(),
            "full_history_dict": _stats(evidence_df), "recent_three_years_dict": _stats(recent_df), "latest_year_dict": _stats(latest_year_df),
            "parking_dict": {symbol_str: {"full_history_dict": _stats(evidence_df.loc[evidence_df["asset_str"].eq(symbol_str)]),
                "recent_three_years_dict": _stats(recent_df.loc[recent_df["asset_str"].eq(symbol_str)]),
                "latest_year_dict": _stats(latest_year_df.loc[latest_year_df["asset_str"].eq(symbol_str)])} for symbol_str in ("BIL", "SPMO")
                if evidence_df["asset_str"].eq(symbol_str).any()},
            "recent_largest_ten_participation_row_list": recent_df.nlargest(10, "participation_float").to_dict("records"),
            "largest_ten_participation_row_list": evidence_df.nlargest(10, "participation_float").to_dict("records")})
        (audit_root_path_obj / "mr_capacity_screen_v3.json").write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
        print(baseline_path_obj.name, report_dict["strategy_list"][-1]["recent_three_years_dict"], flush=True)
        del pricing_df


if __name__ == "__main__":
    main()
