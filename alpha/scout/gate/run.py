"""Run the identity gate for a gated strategy: Scout spec vs the real engine.

Engine source:
- `fresh` (authoritative): the strategy's `run_variant(show_display_bool=False, save_results_bool=False)` now, on
  today's data. It runs BEFORE Scout, because TAA's engine run refreshes the DTB3 cache that Scout then reads.
- `saved` (default, quick): the newest vanilla-backtest pickle. Scout is cut at the pickle's last date, but its
  data are today's Norgate data, so a split or data revision since the pickle can fail the exact tier (a false
  alarm in split-adjusted units): re-run with `--fresh` before trusting a saved-mode failure.

The gate is the exact parity tier (`identity.compare_exact`). The tolerance tier (`identity.compare`) is printed
as information only.
"""

from __future__ import annotations

import functools
import importlib
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate
from alpha.scout.gate.identity import GateReport, compare, compare_exact
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.specs import ndx_vxn
from alpha.scout.specs.ndx_vxn import NDX_VARIANT_DICT

COST_MODEL = CostModel()  # module-level so tests can plant a deliberately wrong cost model


@dataclass(frozen=True)
class GatedSpec:
    name_str: str
    strategy_import_str: str
    pickle_glob_str: str
    run_scout_fn: Callable[[float], tuple[WeightsResult, pd.DataFrame]]  # capital -> (result, close panel)


def _run_taa(variant_name_str: str, capital_float: float):
    from alpha.scout.specs import taa_3x

    config = taa_3x.VARIANT_DICT[variant_name_str].config
    inputs = taa_3x.load_inputs(config=config)
    weight_df = taa_3x.rebalance_weight_df(inputs, config)
    result = simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
        capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=COST_MODEL,
    )
    return result, inputs.close_df


def _taa_gated_spec(variant_name_str: str) -> GatedSpec:
    from alpha.scout.specs.taa_3x import VARIANT_DICT

    import_str = VARIANT_DICT[variant_name_str].strategy_import_str
    module_str = import_str.rsplit(".", 1)[-1]
    return GatedSpec(
        variant_name_str, import_str, f"results/research/strategy/{module_str}/vanilla_backtest/*/{module_str}.pkl",
        functools.partial(_run_taa, variant_name_str),
    )


def _ndx_runner(variant_name_str: str) -> Callable[[float], tuple[WeightsResult, pd.DataFrame]]:
    def run_fn(capital_float: float):
        inputs = ndx_vxn.load_inputs()
        weight_df = ndx_vxn.rebalance_weight_df(inputs, NDX_VARIANT_DICT[variant_name_str].config)
        stock_list = list(weight_df.columns)
        result = simulate(
            inputs.open_df[stock_list], inputs.close_df[stock_list], inputs.dividend_df[stock_list], weight_df,
            start_date=ndx_vxn.TRADING_START_STR, capital_float=capital_float, share_unit_mode_str="historical",
            unadjusted_close_df=inputs.raw_close_df[stock_list], cost_model=COST_MODEL,
        )
        return result, inputs.close_df[stock_list]

    return run_fn


def _ndx_gated_spec(variant_name_str: str) -> GatedSpec:
    strategy_module_str = NDX_VARIANT_DICT[variant_name_str].strategy_module_str
    strategy_name_str = strategy_module_str.rsplit(".", 1)[-1]
    return GatedSpec(
        variant_name_str, strategy_module_str,
        f"results/research/strategy/{strategy_name_str}/vanilla_backtest/*/{strategy_name_str}.pkl",
        _ndx_runner(variant_name_str),
    )


def _run_compass(variant_name_str: str, capital_float: float):
    from alpha.scout.specs import compass

    config = compass.VARIANT_DICT[variant_name_str].config
    inputs = compass.load_inputs(config=config)
    weight_df = compass.rebalance_weight_df(inputs, config)
    result = simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
        capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=compass.ENGINE_COST_MODEL,
    )
    return result, inputs.close_df


def _compass_gated_spec(variant_name_str: str) -> GatedSpec:
    from alpha.scout.specs.compass import VARIANT_DICT

    import_str = VARIANT_DICT[variant_name_str].strategy_import_str
    module_str = import_str.rsplit(".", 1)[-1]
    return GatedSpec(
        variant_name_str, import_str, f"results/research/strategy/{module_str}/vanilla_backtest/*/{module_str}.pkl",
        functools.partial(_run_compass, variant_name_str),
    )


GATED_SPEC_DICT = {
    # The NDX momentum siblings (ndx_vxn is the LIVE pod); see alpha/scout/specs/ndx_vxn.py.
    **{name_str: _ndx_gated_spec(name_str) for name_str in NDX_VARIANT_DICT},
}
# TAA family: the LIVE pod and the engine variants the spec reproduces exactly (alpha/scout/specs/taa_3x.py).
GATED_SPEC_DICT.update({name_str: _taa_gated_spec(name_str) for name_str in (
    "taa_3x", "taa_3x_1n", "taa_lin_1n_qqq", "taa_2x_1n_qld", "taa_nobtal_2x_1n_qld", "taa_nobtal_2x_1n_sso",
)})
# Inflation Compass and its QQQ variant (PM_READY; alpha/scout/specs/compass.py), at the engine's own costs.
GATED_SPEC_DICT.update({name_str: _compass_gated_spec(name_str) for name_str in ("compass", "compass_qqq")})


def _run_core5(capital_float: float):
    from alpha.scout.specs import core5

    inputs = core5.load_inputs()
    result = core5.simulate_config(inputs, core5.LIVE_CONFIG, cost_model=COST_MODEL, capital_float=capital_float)
    return result, inputs.close_df[list(core5.TRADED_TUPLE)]


# CORE5 adaptive macro (PM_READY): five absolute-trend sleeves, BIL and a DBC short (alpha/scout/specs/core5.py).
GATED_SPEC_DICT["core5"] = GatedSpec(
    "core5", "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5",
    "results/research/strategy/strategy_taa_adaptive_macro_core5/vanilla_backtest/*/strategy_taa_adaptive_macro_core5.pkl",
    _run_core5,
)


def _run_beyond_6040_spec(spec_module_str: str, capital_float: float):
    """TFI and Trinity: the spec simulates with the engine's own cost model (their engines differ from CostModel())."""
    spec_module = importlib.import_module(f"alpha.scout.specs.{spec_module_str}")
    inputs = spec_module.load_inputs()
    result = spec_module.simulate_config(inputs, spec_module.LIVE_CONFIG, spec_module.ENGINE_COST_MODEL, capital_float)
    return result, inputs.close_df


def _beyond_6040_gated_spec(name_str: str, spec_module_str: str, strategy_import_str: str) -> GatedSpec:
    module_str = strategy_import_str.rsplit(".", 1)[-1]
    return GatedSpec(
        name_str, strategy_import_str, f"results/research/strategy/{module_str}/vanilla_backtest/*/{module_str}.pkl",
        functools.partial(_run_beyond_6040_spec, spec_module_str),
    )


# PM_READY "beyond 60/40" pods (2026-10-01): alpha/scout/specs/tfi.py and alpha/scout/specs/trinity.py.
GATED_SPEC_DICT.update({
    "tfi": _beyond_6040_gated_spec("tfi", "tfi", "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd"),
    "trinity": _beyond_6040_gated_spec("trinity", "trinity", "strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil"),
})


def _run_sector_ibs(capital_float: float):
    from alpha.scout.specs import sector_ibs

    inputs = sector_ibs.load_inputs()
    return sector_ibs.simulate_config(inputs, sector_ibs.LIVE_CONFIG, sector_ibs.ENGINE_COST_MODEL, capital_float), inputs.close_df


def _run_dispersion_ibs(variant_name_str: str, capital_float: float):
    from alpha.scout.specs import sector_dispersion_ibs

    inputs = sector_dispersion_ibs.load_inputs(variant_name_str)
    config = sector_dispersion_ibs.VARIANT_DICT[variant_name_str].config
    return sector_dispersion_ibs.simulate_config(inputs, config, sector_dispersion_ibs.ENGINE_COST_MODEL, capital_float), inputs.close_df


def _pickle_glob_str(strategy_import_str: str) -> str:
    module_str = strategy_import_str.rsplit(".", 1)[-1]
    return f"results/research/strategy/{module_str}/vanilla_backtest/*/{module_str}.pkl"


# PM_READY sector ETF IBS event pods (2026-10-02): alpha/scout/specs/sector_ibs.py and sector_dispersion_ibs.py, each at
# its engine's own costs (fractional entries, held positions never resized).
GATED_SPEC_DICT["sector_ibs_vox_iyr"] = GatedSpec(
    "sector_ibs_vox_iyr", "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr",
    _pickle_glob_str("strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"), _run_sector_ibs,
)
for _name_str, _import_str in (
    ("dispersion_ibs_kie_ihi_xlc", "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc"),
    ("dispersion_ibs_kie_ihi_xlc_sma200", "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200"),
    ("dispersion_ibs_kie_ihi_sma200", "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200"),
):
    GATED_SPEC_DICT[_name_str] = GatedSpec(_name_str, _import_str, _pickle_glob_str(_import_str), functools.partial(_run_dispersion_ibs, _name_str))


def _engine_strategy(spec: GatedSpec, fresh_bool: bool, root_path: Path):
    if fresh_bool:
        module = importlib.import_module(spec.strategy_import_str)
        return module.run_variant(show_display_bool=False, save_results_bool=False), "fresh run_variant"
    path_list = sorted(root_path.glob(spec.pickle_glob_str))
    if not path_list:
        raise FileNotFoundError(f"No saved engine run for {spec.name_str}; use fresh mode.")
    with path_list[-1].open("rb") as file_obj:
        engine_obj = pickle.load(file_obj)
    saved_str = datetime.fromtimestamp(path_list[-1].stat().st_mtime, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    return engine_obj, f"saved {path_list[-1].relative_to(root_path).as_posix()} (written {saved_str})"


def _daily_weight_df(result: WeightsResult, close_df: pd.DataFrame) -> pd.DataFrame:
    position_df = result.daily_position_df
    marked_df = position_df * close_df.reindex(index=position_df.index, columns=position_df.columns)
    return marked_df.div(result.total_value_ser, axis=0).fillna(0.0)


def run_gate(name_str: str, fresh_bool: bool = False, root_path: Path = MAIN_CHECKOUT_ROOT_PATH) -> GateReport:
    spec = GATED_SPEC_DICT[name_str]
    engine_obj, source_str = _engine_strategy(spec, fresh_bool, root_path)
    capital_float = float(getattr(engine_obj, "_capital_base", 100_000.0))
    engine_results_df = engine_obj.results.copy()
    engine_results_df.index = pd.DatetimeIndex(engine_results_df.index)
    engine_total_ser = engine_results_df["total_value"].astype(float)
    engine_return_ser = engine_total_ser / engine_total_ser.shift(1).fillna(capital_float) - 1.0
    if len(getattr(engine_obj, "realized_weight_df", ())) == 0 and getattr(engine_obj, "_realized_weight_snapshot_row_dict_list", None):
        # Beyond6040Strategy.summarize() skips this step; the daily snapshots are recorded all the same (read-only use).
        engine_obj._materialize_realized_weight_df()
    engine_weight_df = engine_obj.realized_weight_df.drop(columns=["Cash"], errors="ignore")
    engine_weight_df.index = pd.DatetimeIndex(engine_weight_df.index)
    transaction_df = engine_obj.get_transactions() if hasattr(engine_obj, "get_transactions") else engine_obj._transactions
    engine_trade_date_index = pd.DatetimeIndex(pd.to_datetime(transaction_df["bar"]).unique())

    started_float = time.time()
    scout_result, close_df = spec.run_scout_fn(capital_float)
    scout_seconds_float = time.time() - started_float
    engine_end = engine_total_ser.index[-1]
    # *** CRITICAL*** Scout's data may be newer than a saved engine run: compare only up to the engine's last date.
    scout_return_ser = scout_result.daily_return_ser.loc[:engine_end]
    scout_weight_df = _daily_weight_df(scout_result, close_df).loc[:engine_end]
    scout_trade_date_index = pd.DatetimeIndex(pd.to_datetime(scout_result.trade_df["date"]).unique())

    report = compare_exact(
        engine_return_ser, scout_return_ser, engine_weight_df, scout_weight_df, engine_trade_date_index, scout_trade_date_index
    )
    tolerance_report = compare(engine_return_ser, scout_return_ser)
    report.note_list.append(
        "Tolerance tier (information): "
        + "; ".join(f"{name} {check['value']}" for name, check in tolerance_report.check_dict.items())
    )
    report.note_list.append(
        f"Engine source: {source_str}; engine {engine_total_ser.index[0].date()} to {engine_end.date()}, capital {capital_float:,.0f}. "
        f"Scout ran in {scout_seconds_float:.1f}s."
    )
    return report
