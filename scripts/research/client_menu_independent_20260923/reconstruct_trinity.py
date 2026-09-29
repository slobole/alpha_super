"""Recover missing historical holdings only if they reconcile to saved native NAV.

Never change old exports or infer allocation weights from target weights.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from scripts.research.client_menu_independent_20260923.protocol import ROOT_PATH, SOURCE_PATH, STUDY_PATH, digest_str, write_json
from scripts.research.portfolio_family_20260923 import analyze as ledger


def main() -> None:
    source_path = SOURCE_PATH/"data/strategy_taa_trinity_vol_control_8_bil"
    output_path = STUDY_PATH/"trinity_holdings_recovery"
    output_path.mkdir(parents=True, exist_ok=True)
    specification_path = output_path/"spec.json"
    if not specification_path.exists():
        write_json(specification_path, {"frozen_at": datetime.now(timezone.utc).isoformat(), "purpose": "Recover factual saved holdings absent from old export; no new portfolio or strategy search", "prior_results_seen": True, "formula": "q_asset,t=sum executions through t; value_asset,t=q_asset,t*Close_asset,t; native_asset_value=sum values; weights=value/nativeNAV", "timing": "same-date executed quantities and closes for reporting only; no signals, no return filling", "input_files": [{"path_str": str(source_path/file_str), "sha256_str": digest_str(source_path/file_str)} for file_str in ["nav.csv.gz", "transactions.csv.gz", "source_metadata.json"]], "price_asof": "current Norgate data, saved to an immutable study-local parquet", "acceptance": "Every native date present and max absolute asset-value/NAV reconciliation error<=1e-8. Otherwise retain unknown-holdings limitation; no patching old returns."})
    specification_dict = json.loads(specification_path.read_text())
    for input_dict in specification_dict["input_files"]:
        ledger.verify_file(input_dict)
    price_path = output_path/"price_snapshot.parquet"
    if not price_path.exists():
        from strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil import DEFAULT_CONFIG, get_beyond_6040_data
        price_df = get_beyond_6040_data(config=replace(DEFAULT_CONFIG, end_date_str="2026-08-18"))
        price_df.to_parquet(price_path)
    else:
        price_df = pd.read_parquet(price_path)
    nav_df = ledger.read_frame(source_path/"nav.csv.gz", True)
    transaction_df = ledger.read_frame(source_path/"transactions.csv.gz")
    transaction_df["bar"] = pd.to_datetime(transaction_df.bar)
    quantity_change_df = transaction_df.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum")
    # *** CRITICAL *** No trade on an existing native session -> zero QUANTITY
    # CHANGE. This never fills prices, NAV returns, or an absent session.
    quantity_change_df = quantity_change_df.reindex(nav_df.index, fill_value=0.).fillna(0.)
    quantity_df = quantity_change_df.cumsum()
    close_df = price_df.loc[nav_df.index, [(asset_str, "Close") for asset_str in quantity_df.columns]].copy()
    close_df.columns = quantity_df.columns
    if not np.isfinite(close_df.to_numpy()).all():
        raise ValueError("Cannot recover holdings using missing closes")
    value_df = quantity_df*close_df
    error_series = (value_df.sum(axis=1)-nav_df.portfolio_value).abs()/nav_df.total_value
    passed_bool = bool(error_series.max() <= 1e-8)
    weight_df = value_df.div(nav_df.total_value, axis=0)
    weight_df["Cash"] = nav_df.cash/nav_df.total_value
    holdings_path = output_path/"realized_weights.csv.gz"
    if passed_bool:
        if not np.allclose(weight_df.sum(axis=1), 1., atol=1e-8, rtol=0.):
            raise AssertionError("Recovered holdings and cash do not sum to native NAV")
        weight_df.to_csv(holdings_path)
    manifest_dict = {"status": "reconciled" if passed_bool else "unresolved", "max_relative_nav_error": float(error_series.max()), "dates": len(nav_df), "start": str(nav_df.index[0].date()), "end": str(nav_df.index[-1].date()), "price_sha256": digest_str(price_path), "price_path": str(price_path), "price_attrs": str(price_df.attrs), "holdings_sha256": digest_str(holdings_path) if passed_bool else None, "holdings_path": str(holdings_path) if passed_bool else None, "code_sha256": digest_str(__import__("pathlib").Path(__file__)), "spec_sha256": digest_str(specification_path)}
    write_json(output_path/"manifest.json", manifest_dict)
    print(json.dumps({key_str: manifest_dict[key_str] for key_str in ["status", "max_relative_nav_error", "dates"]}))


if __name__ == "__main__":
    main()
