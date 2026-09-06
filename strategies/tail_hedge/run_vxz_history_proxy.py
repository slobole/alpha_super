"""Isolated original-VXZ ETN diagnostic; never a VIXM history backfill."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha.engine.backtest import run_daily
from data.norgate_loader import load_raw_prices
from strategies.tail_hedge import strategy_vixm_backwardation as vixm_module
from strategies.tail_hedge.run_tail_hedge_vanilla_study import nav_return_ser


def trim_proxy_boundaries(price_df: pd.DataFrame) -> pd.DataFrame:
    close_ser = price_df[("VIXM", "Close")]
    if close_ser.first_valid_index() is None:
        raise ValueError("No observed ETN history.")
    # *** CRITICAL*** Trim inception/retirement only, never remove internal P&L.
    bounded_df = price_df.loc[close_ser.first_valid_index():close_ser.last_valid_index()].copy()
    for asset_str in ("VIXM", "SHY", "$SPX"):
        if bounded_df[(asset_str, "Close")].isna().any():
            raise ValueError("Missing internal proxy/benchmark close session.")
    return bounded_df


def run_proxy(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    config_obj = replace(vixm_module.DEFAULT_CONFIG,
        strategy_name_str="diagnostic_original_vxz_backwardation",
        history_start_date_str="2009-01-01", backtest_start_date_str="2009-02-02",
        end_date_str="2019-01-28")
    raw_price_df = load_raw_prices(symbols=["VXZ-201901", "SHY"],
        benchmarks=["$SPX", "$VIX", "$VIX3M"],
        start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
    raw_price_df.to_csv(output_path / "authentic_instrument_input.csv.gz")
    # *** CRITICAL*** Explicit engine-only symbol alias, never a time splice.
    # Every VIXM-labelled execution bar in this isolated run is original VXZ.
    # Signal closes are contemporaneous T; frozen engine fills at Open_(T+1).
    proxy_price_df = raw_price_df.loc[:, raw_price_df.columns.get_level_values(0).isin(
        ["VXZ-201901", "SHY", "$SPX"])].copy()
    proxy_price_df.columns = pd.MultiIndex.from_tuples([
        ("VIXM" if asset_str == "VXZ-201901" else asset_str, field_str)
        for asset_str, field_str in proxy_price_df.columns])
    proxy_price_df[(vixm_module.VIX_SIGNAL_NAMESPACE_STR, "vix_close_float")] = raw_price_df[("$VIX", "Close")]
    proxy_price_df[(vixm_module.VIX_SIGNAL_NAMESPACE_STR, "vix3m_close_float")] = raw_price_df[("$VIX3M", "Close")]
    proxy_price_df = trim_proxy_boundaries(proxy_price_df)
    proxy_price_df.attrs.update(raw_price_df.attrs)
    proxy_price_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "VIXM": "CAPITALSPECIAL", "SHY": "CAPITALSPECIAL", "$SPX": "TOTALRETURN",
        vixm_module.VIX_SIGNAL_NAMESPACE_STR: "RAW_INDEX_CLOSE"}
    strategy_obj = vixm_module.VixmBackwardationStrategy(config_obj)
    calendar_idx = vixm_module.build_execution_calendar_idx(proxy_price_df, config_obj.backtest_start_date_str)
    run_daily(strategy_obj, proxy_price_df, calendar=calendar_idx,
        show_progress=False, show_signal_progress_bool=False,
        audit_override_bool=True, audit_sample_size_int=10)
    strategy_obj.results.to_csv(output_path / "VXZ_proxy_results.csv")
    return_ser = nav_return_ser(strategy_obj.results.total_value, config_obj.capital_base_float)
    return_ser.rename("VXZ_proxy").to_csv(output_path / "returns.csv")
    transaction_df = strategy_obj.get_transactions().copy()
    transaction_df["asset"] = transaction_df.asset.replace({"VIXM": "VXZ-201901"})
    transaction_df.to_csv(output_path / "authentic_instrument_transactions.csv", index=False)
    strategy_obj.signal_diagnostic_df.to_csv(output_path / "signals.csv")
    manifest_dict = {"scope": "research_proxy_only", "instrument": "VXZ-201901",
        "norgate_asset_id": 442125, "engine_internal_alias": "VIXM",
        "alias_is_not_actual_vixm": True, "timing": "Close_T to Open_T+1",
        "slippage_per_side": .001, "positive_dividend_withholding": 0.,
        "capital": config_obj.capital_base_float,
        "start": str(return_ser.index[0]), "end": str(return_ser.index[-1]),
        "input_sha256": hashlib.sha256((output_path / "authentic_instrument_input.csv.gz").read_bytes()).hexdigest(),
        "strategy_source_sha256": hashlib.sha256(Path(vixm_module.__file__).read_bytes()).hexdigest(),
        "proxy_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caveat": "Original ETN, fees and issuer credit differ. No splice, no 2008 claim."}
    (output_path / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    print(manifest_dict, flush=True)


if __name__ == "__main__":
    run_proxy(Path("results/research/proxy_tail_architecture_20260905/vxz_proxy_verified"))
