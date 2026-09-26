"""Frozen four-arm damage attribution; one new NATR hypothesis, no tuning."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine import strategy as engine_strategy_module
from strategies.momentum import strategy_mo_atr_normalized_ndx as base_module
from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module
from strategies.momentum import strategy_mo_mosaic_russell1000 as mosaic_module


VARIANT_TUPLE = (
    "legacy_vintage_diagnostic", "asof_atr_signal_only_attribution",
    "asof_atr_corrected", "natr20_corrected",
)
TURNOVER_ONLY_VARIANT_STR = "asof_atr_turnover_only_attribution"
ORIGINAL_SIGNAL_FUNCTION = base_module.compute_atr_normalized_signal_tables


class ImmutableStringMap(Mapping):
    """Study-only provenance map: immutable content need not be deep-copied."""

    __slots__ = ("_mapping_obj",)

    def __init__(self, value_dict):
        if not all(isinstance(key_str, str) and isinstance(value_str, str)
                   for key_str, value_str in value_dict.items()):
            raise TypeError("Only flat string-to-string provenance is supported.")
        object.__setattr__(self, "_mapping_obj", MappingProxyType(dict(value_dict)))

    def __getitem__(self, key_str):
        return self._mapping_obj[key_str]

    def __iter__(self):
        return iter(self._mapping_obj)

    def __len__(self):
        return len(self._mapping_obj)

    def __setattr__(self, key_str, value_obj):
        raise TypeError("Study provenance metadata is immutable.")

    def __delattr__(self, key_str):
        raise TypeError("Study provenance metadata is immutable.")

    def __deepcopy__(self, memo_dict):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def _plain_metadata_obj(value_obj):
    if isinstance(value_obj, Mapping):
        return {key_str: _plain_metadata_obj(item_obj) for key_str, item_obj in value_obj.items()}
    if isinstance(value_obj, (list, tuple)):
        return [_plain_metadata_obj(item_obj) for item_obj in value_obj]
    return value_obj


def provider_metadata_digest_str(metadata_dict):
    metadata_bytes = json.dumps(
        _plain_metadata_obj(metadata_dict), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(metadata_bytes).hexdigest()


def make_study_metadata_immutable(pricing_df):
    """Preserve every provider attribute; replace only its flat adjustment map."""
    original_digest_str = provider_metadata_digest_str(pricing_df.attrs)
    adjustment_metadata_dict = pricing_df.attrs["norgate_adjustment_by_symbol_dict"]
    immutable_metadata_obj = ImmutableStringMap(adjustment_metadata_dict)
    if dict(immutable_metadata_obj) != dict(adjustment_metadata_dict):
        raise RuntimeError("Study metadata conversion changed adjustment provenance.")
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = immutable_metadata_obj
    if provider_metadata_digest_str(pricing_df.attrs) != original_digest_str:
        raise RuntimeError("Study metadata conversion changed provider attributes.")
    return {
        "immutable_provider_metadata_bool": True,
        "provider_metadata_content_sha256": original_digest_str,
        "provider_adjustment_entry_count": len(immutable_metadata_obj),
    }


def study_source_provenance_dict():
    return {
        "study_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine_strategy_sha256": sha256(Path(engine_strategy_module.__file__).read_bytes()).hexdigest(),
        "immutable_metadata_source_sha256": sha256(inspect.getsource(ImmutableStringMap).encode("utf-8")).hexdigest(),
    }


# Capture before any backtest. Reading mutable source paths after a long run
# could otherwise mislabel an already-running process with later disk edits.
STUDY_SOURCE_PROVENANCE_DICT = study_source_provenance_dict()


def legacy_signal_tables(*argument_tuple, **argument_dict):
    # Diagnostic only: deliberately reproduce the invalid vintage units.
    argument_dict["price_unadjusted_close_df"] = argument_dict["price_close_df"]
    return ORIGINAL_SIGNAL_FUNCTION(*argument_tuple, **argument_dict)


class StudyMixin:
    study_variant_str: str

    def compute_signals(self, pricing_data):
        if self.study_variant_str == "legacy_vintage_diagnostic":
            with patch.object(base_module, "compute_atr_normalized_signal_tables", legacy_signal_tables):
                signal_df = super().compute_signals(pricing_data)
        else:
            signal_df = super().compute_signals(pricing_data)
        if self.study_variant_str == "natr20_corrected":
            # *** CRITICAL *** Nominal ATR and raw Close are both at decision
            # T. (ROC/ATR_nominal)*rawClose = ROC/(ATR_adjusted/adjustedClose).
            for symbol_str in self.get_tradeable_symbol_list(pricing_data):
                signal_df[(symbol_str, "risk_adj_score_ser")] *= pricing_data[(symbol_str, "Unadjusted Close")]
        if self.study_variant_str in VARIANT_TUPLE[:2] and hasattr(self, "dollar_adv_df"):
            # Retain original mixed-unit liquidity only in attribution arms.
            legacy_volume_df = pd.DataFrame({
                symbol_str: pricing_data[(symbol_str, "Unadjusted Close")] * pricing_data[(symbol_str, "Volume")]
                for symbol_str in self.get_tradeable_symbol_list(pricing_data)
            })
            self.dollar_adv_df = legacy_volume_df.rolling(self.adv_window_int, min_periods=self.adv_window_int).median()
        return signal_df

    def get_target_weight_ser(self, close_row_ser):
        target_ser = super().get_target_weight_ser(close_row_ser)
        self.decision_record_list.append({
            "decision_date": str(pd.Timestamp(self.previous_bar).date()),
            "execution_date": str(pd.Timestamp(self.current_bar).date()),
            "weights": {str(symbol_str): float(weight_float) for symbol_str, weight_float in target_ser.items()},
        })
        return target_ser


class StudyNdx(StudyMixin, base_module.AtrNormalizedNdxStrategy):
    pass


class StudyVxn(StudyMixin, vxn_module.VxnScaledAtrNormalizedNdxStrategy):
    pass


class StudyMosaic(StudyMixin, mosaic_module.MosaicRussell1000Strategy):
    pass


def build_strategy(family_str, variant_str, input_tuple):
    config_obj = {"ndx": base_module.DEFAULT_CONFIG, "vxn": vxn_module.DEFAULT_CONFIG, "mosaic": mosaic_module.DEFAULT_CONFIG}[family_str]
    strategy_class = {"ndx": StudyNdx, "vxn": StudyVxn, "mosaic": StudyMosaic}[family_str]
    argument_dict = dict(
        name=f"{family_str}_{variant_str}", benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=input_tuple[2], regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=100_000.0, slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    if family_str == "vxn":
        argument_dict["vxn_scale_signal_df"] = input_tuple[3]
    if family_str == "mosaic":
        argument_dict.update(
            corr_window_int=config_obj.corr_window_int, corr_min_overlap_int=config_obj.corr_min_overlap_int,
            corr_penalty_lambda_float=config_obj.corr_penalty_lambda_float,
            min_dollar_adv_float=config_obj.min_dollar_adv_float, adv_window_int=config_obj.adv_window_int,
        )
    strategy_obj = strategy_class(**argument_dict)
    strategy_obj.study_variant_str = variant_str
    strategy_obj.historical_share_units_bool = variant_str in ("asof_atr_corrected", "natr20_corrected")
    strategy_obj.decision_record_list = []
    strategy_obj.universe_df = input_tuple[1]
    base_module.configure_total_return_benchmark_provenance(strategy_obj, config_obj)
    return strategy_obj


def metric_dict(equity_ser, initial_float):
    equity_ser = equity_ser.astype(float)
    return_ser = equity_ser.pct_change(fill_method=None)
    return_ser.iloc[0] = equity_ser.iloc[0] / initial_float - 1.0
    years_float = len(return_ser) / 252.0
    cagr_float = (equity_ser.iloc[-1] / initial_float) ** (1.0 / years_float) - 1.0
    vol_float = float(return_ser.std(ddof=1) * np.sqrt(252.0))
    maxdd_float = float((equity_ser / equity_ser.cummax().clip(lower=initial_float) - 1.0).min())
    return dict(cagr=float(cagr_float), volatility=vol_float,
        sharpe=float(return_ser.mean() * 252.0 / vol_float), max_drawdown=maxdd_float,
        mar=float(cagr_float / abs(maxdd_float)), final_equity=float(equity_ser.iloc[-1]))


def run_family(family_str, input_root_path, output_root_path, variant_list):
    cache_path = input_root_path / ("mosaic_inputs.pkl" if family_str == "mosaic" else "ndx_inputs.pkl")
    input_tuple = pd.read_pickle(cache_path)
    price_df = input_tuple[0]
    # Identical metadata-only optimization in every attribution arm. Provider
    # content is retained; this does not change prices, PIT membership or timing.
    metadata_provenance_dict = make_study_metadata_immutable(price_df)
    source_provenance_dict = dict(STUDY_SOURCE_PROVENANCE_DICT)
    calendar_idx = price_df.index[price_df.index >= pd.Timestamp("2000-01-01")]
    for variant_str in variant_list:
        output_path = output_root_path / family_str / variant_str
        output_path.mkdir(parents=True, exist_ok=True)
        strategy_obj = build_strategy(family_str, variant_str, input_tuple)
        print(f"START {family_str} {variant_str}", flush=True)
        run_daily(strategy_obj, price_df, calendar=calendar_idx, show_progress=False,
            show_signal_progress_bool=False, audit_override_bool=False)
        result_df = strategy_obj.results.copy()
        result_df.index = pd.to_datetime(result_df.index)
        result_df.to_csv(output_path / "daily_results.csv")
        transaction_df = strategy_obj.get_transactions().copy()
        transaction_df.to_csv(output_path / "transactions.csv", index=False)
        (output_path / "decisions.json").write_text(json.dumps(strategy_obj.decision_record_list, indent=2), encoding="utf-8")
        strategy_obj.summary.to_csv(output_path / "engine_summary.csv")
        result_dict = metric_dict(result_df["total_value"], 100_000.0)
        result_dict.update(family=family_str, variant=variant_str,
            start=str(result_df.index.min().date()), end=str(result_df.index.max().date()),
            transactions=len(transaction_df), commissions=float(transaction_df["commission"].sum()),
            min_cash=float(result_df["cash"].min()),
            mean_positions=float(np.mean([len(row_dict["weights"]) for row_dict in strategy_obj.decision_record_list])),
            mean_target_exposure=float(np.mean([sum(row_dict["weights"].values()) for row_dict in strategy_obj.decision_record_list])),
            accounting_policy=dict(strategy_obj._accounting_policy_dict),
            historical_share_units_bool=bool(strategy_obj.historical_share_units_bool),
            **source_provenance_dict,
            **metadata_provenance_dict,
            configuration={key_str: value_obj for key_str, value_obj in argument_config_dict(family_str).items()},
        )
        if len(transaction_df):
            transaction_date_idx = pd.to_datetime(transaction_df["bar"])
            prior_equity_ser = result_df["total_value"].shift(1)
            prior_equity_ser.iloc[0] = 100_000.0
            trade_notional_ser = transaction_df["total_value"].abs().astype(float)
            denominator_vec = prior_equity_ser.reindex(transaction_date_idx).to_numpy()
            result_dict["annual_oneway_turnover"] = float((trade_notional_ser.to_numpy() / denominator_vec).sum() / 2.0 / (len(result_df) / 252.0))
            result_dict["slippage_cash_proxy"] = float(trade_notional_ser.sum() * strategy_obj._slippage)
        period_list = []
        for start_str, end_str in [("2000-01-01", "2012-12-31"), ("2013-01-01", "2019-12-31"), ("2020-01-01", "2026-09-25")]:
            period_ser = result_df.loc[start_str:end_str, "total_value"]
            earlier_ser = result_df.loc[result_df.index < pd.Timestamp(start_str), "total_value"]
            initial_float = float(earlier_ser.iloc[-1]) if len(earlier_ser) else 100_000.0
            period_list.append(dict(start=start_str, end=end_str, **metric_dict(period_ser, initial_float)))
        result_dict["subperiods"] = period_list
        (output_path / "metrics.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result_dict), flush=True)
        del strategy_obj


def argument_config_dict(family_str):
    from dataclasses import asdict
    config_obj = {"ndx": base_module.DEFAULT_CONFIG, "vxn": vxn_module.DEFAULT_CONFIG, "mosaic": mosaic_module.DEFAULT_CONFIG}[family_str]
    config_dict = asdict(config_obj)
    config_dict["end_date_str"] = "2026-09-25"
    return config_dict


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--family", choices=["ndx", "vxn", "mosaic"], required=True)
    parser_obj.add_argument("--inputs", type=Path, required=True)
    parser_obj.add_argument("--outputs", type=Path, required=True)
    parser_obj.add_argument("--variants", nargs="+", choices=(*VARIANT_TUPLE, TURNOVER_ONLY_VARIANT_STR), default=list(VARIANT_TUPLE))
    argument_obj = parser_obj.parse_args()
    run_family(argument_obj.family, argument_obj.inputs, argument_obj.outputs, argument_obj.variants)
