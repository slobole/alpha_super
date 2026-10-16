"""Small independent native-stock prefix/split checks; no provider or broker IO.

Uses three stock histories per parent feature implementation, with real held
names when a trace exists. This is feature evidence, not full-universe decision
parity, investment validation or a search for a better strategy.
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


def _new_strategy(family_str):
    if family_str == "hpi":
        from strategies.hpi.stateful_long import HPIStatefulLongStrategy, ENTRY_HORIZON_VOTE_STR, TURNOVER_FIELD_STR
        return HPIStatefulLongStrategy(name="mr_native_feature_audit", benchmarks=[], capital_base=1_000_000,
            entry_mode_str=ENTRY_HORIZON_VOTE_STR, ranking_field_str=TURNOVER_FIELD_STR)
    if family_str == "dv2_ind":
        from strategies.dv2.strategy_mr_dv2_industry_etf import DVO2IndustryEtfStrategy
        return DVO2IndustryEtfStrategy(name="mr_native_feature_audit", benchmarks=[], capital_base=1_000_000)
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
    return DVO2Strategy(name="mr_native_feature_audit", benchmarks=[], capital_base=1_000_000)


def _entry_mask(feature_df, symbol_str, family_str):
    if family_str == "hpi":
        vote_vec = sum((feature_df[(symbol_str, return_field_str)] < 0).astype(int)
            * (feature_df[(symbol_str, hpi_field_str)] < 30).astype(int)
            for return_field_str, hpi_field_str in (("return_2d_ser", "hpi_2d_ser"),
                ("return_3d_ser", "hpi_value_ser"), ("return_5d_ser", "hpi_5d_ser")))
        return (vote_vec >= 2) & (feature_df[(symbol_str, "ibs_value_ser")] < .10) & (
            feature_df[(symbol_str, "Close")] > feature_df[(symbol_str, "sma_200_price_ser")])
    return (feature_df[(symbol_str, "dv2")] < 10) & (feature_df[(symbol_str, "p126d_return")] > .05) & (
        feature_df[(symbol_str, "Close")] > feature_df[(symbol_str, "sma_200")])


def main():
    audit_root_path_obj = REPO_PATH_OBJ / "results/review/wired_signoff_20261015"
    report_dict = {"generated_at_utc_str": datetime.now(timezone.utc).isoformat(), "new_strategy_variants_int": 0,
        "scope_str": "Three native stock histories per parent feature family; full loaded past retained; no provider/broker IO.",
        "family_list": [], "limitations_list": ["This samples native symbols, not all potential selections.",
            "Split transformations preserve native dtype; observed threshold flips may reflect native precision and need economic materiality assessment.",
            "Entry predicate excludes PIT/liquidity eligibility and cross-sectional ranking; those have separate actual-host checks."]}
    family_tuple = (("dv2", "strategy_mr_dv2"), ("hpi", "strategy_mr_hpi_sp500_2_3_5_vote"),
                    ("dv2_ind", "strategy_mr_dv2_industry_etf"))
    for family_str, module_str in family_tuple:
        input_path_obj = audit_root_path_obj / "input_preparation" / module_str / "pricing.parquet"
        pricing_df = pd.read_parquet(input_path_obj)
        auxiliary_dict = pd.read_pickle(input_path_obj.parent / "auxiliary.pkl.gz")
        universe_df = auxiliary_dict["universe_df"]
        symbol_list = sorted(str(symbol_str) for symbol_str in pricing_df.columns.get_level_values(0).unique()
                             if not str(symbol_str).startswith("$") and symbol_str not in ("BIL", "SPMO"))
        trace_path_obj = audit_root_path_obj / "baselines" / module_str / "trace.pkl.gz"
        held_symbol_list = []
        if trace_path_obj.exists():
            trace_list = pd.read_pickle(trace_path_obj)
            held_symbol_list = sorted({symbol_str for trace_row_dict in trace_list[-40:]
                                      for symbol_str in trace_row_dict["positions"] if symbol_str in symbol_list})
        selected_symbol_list = list(dict.fromkeys([*held_symbol_list[:2], *symbol_list]))[:3]
        sample_df = pricing_df.loc[:, [column_tuple for column_tuple in pricing_df.columns if column_tuple[0] in selected_symbol_list]].copy()
        del pricing_df
        feature_df = _new_strategy(family_str).compute_signals(sample_df)
        family_dict = {"family_str": family_str, "source_price_sha256_str": hashlib.sha256(input_path_obj.read_bytes()).hexdigest(),
            "selected_symbol_list": selected_symbol_list, "real_held_symbol_list": held_symbol_list[:2],
            "prefix_list": [], "split_list": []}
        cutoff_date_list = ["2019-08-30", "2020-03-16", "2022-06-17", "2024-07-03", "2025-03-10", "2026-09-30", "2026-10-01", "2026-10-13"]
        for cutoff_str in cutoff_date_list:
            cutoff_ts = pd.Timestamp(cutoff_str)
            if cutoff_ts not in sample_df.index:
                continue
            # *** CRITICAL *** Every rolling/return/rank computation is repeated
            # on the exact prefix <= Close_T, with no future row or back-fill.
            prefix_feature_df = _new_strategy(family_str).compute_signals(sample_df.loc[:cutoff_ts].copy())
            try:
                pd.testing.assert_series_equal(feature_df.loc[cutoff_ts], prefix_feature_df.loc[cutoff_ts], check_exact=True)
                prefix_pass_bool, error_str = True, None
            except AssertionError as exception_obj:
                prefix_pass_bool, error_str = False, str(exception_obj)[:500]
            leak_symbol_str = next(symbol_str for symbol_str in selected_symbol_list
                if np.isfinite(sample_df[(symbol_str, "Close")].shift(-1).loc[cutoff_ts]))
            # A planted use of Close_(T+1) must fail a row-T prefix comparison.
            leak_full_float = sample_df[(leak_symbol_str, "Close")].shift(-1).loc[cutoff_ts]
            leak_prefix_float = sample_df.loc[:cutoff_ts, (leak_symbol_str, "Close")].shift(-1).loc[cutoff_ts]
            positive_control_caught_bool = np.isfinite(leak_full_float) and pd.isna(leak_prefix_float)
            family_dict["prefix_list"].append({"cutoff_str": cutoff_str, "exact_feature_row_equal_bool": prefix_pass_bool,
                "planted_next_close_leak_caught_bool": bool(positive_control_caught_bool), "error_str": error_str})
        for split_factor_float in (40.0, .1, 1.5):
            split_df = sample_df.copy()
            for symbol_str in selected_symbol_list:
                for field_str in ("Open", "High", "Low", "Close", "Dividend"):
                    if (symbol_str, field_str) in split_df.columns:
                        split_df[(symbol_str, field_str)] = (split_df[(symbol_str, field_str)] / split_factor_float).astype(sample_df[(symbol_str, field_str)].dtype)
                if (symbol_str, "Volume") in split_df.columns:
                    split_df[(symbol_str, "Volume")] = (split_df[(symbol_str, "Volume")] * split_factor_float).astype(sample_df[(symbol_str, "Volume")].dtype)
            split_feature_df = _new_strategy(family_str).compute_signals(split_df)
            flip_row_list = []
            for symbol_str in selected_symbol_list:
                baseline_mask_ser = _entry_mask(feature_df, symbol_str, family_str)
                split_mask_ser = _entry_mask(split_feature_df, symbol_str, family_str)
                # Report every decision-level threshold flip after the strategy's
                # canonical first execution year, not just an allclose feature score.
                flip_mask_ser = baseline_mask_ser.ne(split_mask_ser) & (sample_df.index >= pd.Timestamp("2012-01-03" if family_str == "dv2_ind" else "2004-01-01"))
                for decision_ts in sample_df.index[flip_mask_ser]:
                    member_bool = bool(symbol_str in universe_df.columns and decision_ts in universe_df.index
                        and universe_df.loc[decision_ts, symbol_str] == 1)
                    diagnostic_field_tuple = ("ibs_value_ser", "rsi2_value_ser", "hpi_2d_ser", "hpi_value_ser", "hpi_5d_ser") if family_str == "hpi" else ("dv2", "p126d_return", "natr")
                    flip_row_list.append({"symbol_str": symbol_str, "date_str": decision_ts.date().isoformat(),
                        "baseline_entry_bool": bool(baseline_mask_ser.loc[decision_ts]), "split_entry_bool": bool(split_mask_ser.loc[decision_ts]),
                        "pit_member_at_date_bool": member_bool,
                        "baseline_feature_dict": {field_str: float(feature_df.loc[decision_ts, (symbol_str, field_str)]) for field_str in diagnostic_field_tuple},
                        "split_feature_dict": {field_str: float(split_feature_df.loc[decision_ts, (symbol_str, field_str)]) for field_str in diagnostic_field_tuple}})
            family_dict["split_list"].append({"split_factor_float": split_factor_float,
                "entry_predicate_flip_count_int": len(flip_row_list),
                "pit_eligible_entry_predicate_flip_count_int": sum(row_dict["pit_member_at_date_bool"] for row_dict in flip_row_list),
                "flip_row_list": flip_row_list})
        injection_cutoff_ts = pd.Timestamp("2026-10-13")
        injection_symbol_str = next(symbol_str for symbol_str in selected_symbol_list
            if np.isfinite(sample_df[(symbol_str, "Close")].shift(-1).loc[injection_cutoff_ts]))
        injection_strategy_obj = _new_strategy(family_str)
        original_compute_fn = injection_strategy_obj.compute_signals

        def injected_compute(price_df):
            injected_feature_df = original_compute_fn(price_df)
            # *** CRITICAL *** Deliberate positive-control bug in a real entry
            # feature: Close_(T+1) must be unavailable to a Close_T decision.
            injected_field_str = "return_3d_ser" if family_str == "hpi" else "p126d_return"
            injected_lag_int = 2 if family_str == "hpi" else 125
            injected_feature_df[(injection_symbol_str, injected_field_str)] = (
                price_df[(injection_symbol_str, "Close")].shift(-1)
                / price_df[(injection_symbol_str, "Close")].shift(injected_lag_int) - 1.0)
            return injected_feature_df

        injection_strategy_obj.compute_signals = injected_compute
        full_injected_df = injection_strategy_obj.compute_signals(sample_df.copy())
        prefix_injected_df = injection_strategy_obj.compute_signals(sample_df.loc[:injection_cutoff_ts].copy())
        try:
            pd.testing.assert_series_equal(full_injected_df.loc[injection_cutoff_ts], prefix_injected_df.loc[injection_cutoff_ts], check_exact=True)
            injected_feature_caught_bool = False
        except AssertionError:
            injected_feature_caught_bool = True
        family_dict["strategy_entry_feature_leak_positive_control_dict"] = {
            "symbol_str": injection_symbol_str, "cutoff_str": str(injection_cutoff_ts.date()),
            "feature_field_str": "return_3d_ser" if family_str == "hpi" else "p126d_return",
            "caught_by_actual_feature_row_comparator_bool": injected_feature_caught_bool}
        family_dict["prefix_pass_bool"] = all(row_dict["exact_feature_row_equal_bool"] and row_dict["planted_next_close_leak_caught_bool"] for row_dict in family_dict["prefix_list"])
        family_dict["entry_split_invariance_pass_bool"] = not any(row_dict["entry_predicate_flip_count_int"] for row_dict in family_dict["split_list"])
        report_dict["family_list"].append(family_dict)
        (audit_root_path_obj / "mr_native_invariance_v3.json").write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
        print(f"{family_str}: {len(family_dict['prefix_list'])} prefixes, split flips {sum(row_dict['entry_predicate_flip_count_int'] for row_dict in family_dict['split_list'])}", flush=True)


if __name__ == "__main__":
    main()
