"""A strategy family as Scout runs it: a registered grid around a spec, executed by the parity weights engine.

`FamilyRunner.run_config(config_dict, cost_model, capital_float)` returns the engine result of one configuration;
the grid order is `Registration.grid_config_list` order (parameter names sorted, values in registered order), so
the plateau grid shape is `grid_shape_tuple`. `decision_offset_int` is not a grid axis: it is the luck band (S4).

The two LIVE pods are wired here (P5): TAA 3x and NDX VXN, executed exactly as the identity gate executes them.
The gated NDX siblings (plain ATR, NATR20, NATR20 VXN) reuse the NDX grid around their own engine config.
The gated TAA engine variants (1/N, linearity, 2x) are wired through `taa_variant_family` the same way.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate

DEFAULT_COST_MODEL = CostModel()


@dataclass
class FamilyRunner:
    name_str: str
    family_id_str: str
    param_grid_dict: dict  # name -> tuple of values, in registered order
    live_config_dict: dict
    simulate_fn: Callable[[dict, CostModel, float], WeightsResult]
    offset_count_int: int = 16  # monthly rebalance: offsets 0-15 (a month always has more than 16 sessions)

    @property
    def name_list(self) -> list[str]:
        return sorted(self.param_grid_dict)

    @property
    def grid_shape_tuple(self) -> tuple[int, ...]:
        return tuple(len(self.param_grid_dict[name_str]) for name_str in self.name_list)

    def config_list(self) -> list[dict]:
        value_list_list = [list(self.param_grid_dict[name_str]) for name_str in self.name_list]
        return [dict(zip(self.name_list, combo_tuple)) for combo_tuple in itertools.product(*value_list_list)]

    def label_str(self, config_dict: dict) -> str:
        return "|".join(f"{name_str}={_value_str(config_dict[name_str])}" for name_str in self.name_list)

    def run_config(self, config_dict: dict, cost_model: CostModel = DEFAULT_COST_MODEL, capital_float: float = 100_000.0) -> WeightsResult:
        return self.simulate_fn(config_dict, cost_model, capital_float)


def _value_str(value_obj) -> str:
    if isinstance(value_obj, tuple):
        return "-".join(str(v) for v in value_obj)
    return str(value_obj)


# ---------------------------------------------------------------- TAA 3x
TAA_GRID_DICT = {
    "momentum_month_tuple": ((1, 3), (1, 3, 6), (1, 3, 6, 12), (3, 6, 12), (6, 12)),  # ordered by mean horizon
    "realized_vol_window_int": (10, 20, 40, 63),
}


# The linearity variant replaces the momentum months by regression lookbacks (sessions): the same five horizon sets
# as TAA_GRID_DICT at about 21 sessions a month, ordered by mean horizon, the live set in the middle. The threshold
# (daily log slope x adjusted R2) is centred on the live 0.0; +-0.0001 (about +-2.5% a year of R2-weighted trend) moves
# the share of passing defensive slots from 56% to 65% / 46% (2012-2026 month ends), a material but local change.
TAA_LINEARITY_GRID_DICT = {
    "linearity_day_tuple": ((21, 63), (21, 63, 126), (21, 63, 126, 252), (63, 126, 252), (126, 252)),
    "linearity_threshold_float": (-0.0001, 0.0, 0.0001),
    "realized_vol_window_int": (10, 20, 40, 63),
}
TAA_FAMILY_NAME_DICT = {
    "taa_3x": "TAA 3x",
    "taa_3x_1n": "TAA 3x 1/N",
    "taa_lin_1n_qqq": "TAA linearity 1/N QQQ",
    "taa_2x_1n_qld": "TAA 2x 1/N QLD",
    "taa_nobtal_2x_1n_qld": "TAA no-BTAL 2x 1/N QLD",
    "taa_nobtal_2x_1n_sso": "TAA no-BTAL 2x 1/N SSO",
}


def taa_variant_family(variant_name_str: str, inputs=None) -> FamilyRunner:
    """A gated TAA variant (alpha/scout/specs/taa_3x.py VARIANT_DICT) as a family: grid axes vary around its config."""
    from dataclasses import replace

    from alpha.scout.specs import taa_3x

    base_config = taa_3x.VARIANT_DICT[variant_name_str].config
    inputs = inputs or taa_3x.load_inputs(config=base_config)

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        weight_df = taa_3x.rebalance_weight_df(inputs, replace(base_config, **config_dict))
        return simulate(
            inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
            capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=cost_model,
        )

    grid_dict = TAA_LINEARITY_GRID_DICT if base_config.score_str == "linearity" else TAA_GRID_DICT
    return FamilyRunner(
        name_str=TAA_FAMILY_NAME_DICT[variant_name_str], family_id_str="tactical_asset_allocation", param_grid_dict=grid_dict,
        live_config_dict={name_str: getattr(base_config, name_str) for name_str in grid_dict}, simulate_fn=simulate_fn,
    )


def taa_3x_family(inputs=None) -> FamilyRunner:
    return taa_variant_family("taa_3x", inputs)


# ---------------------------------------------------------------- NDX VXN
NDX_GRID_DICT = {
    "roc_month_int": (6, 9, 12, 15),
    "stock_sma_int": (50, 100, 200),
    "top_count_int": (5, 10, 15),
}


def _ndx_family(variant_name_str: str, name_str: str, inputs=None) -> FamilyRunner:
    """One NDX momentum sibling (alpha/scout/specs/ndx_vxn.py NDX_VARIANT_DICT); the grid moves around its config."""
    from alpha.scout.specs import ndx_vxn

    inputs = inputs or ndx_vxn.load_inputs()
    base_config = ndx_vxn.NDX_VARIANT_DICT[variant_name_str].config

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        weight_df = ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(base_config, **config_dict))
        stock_list = list(weight_df.columns)
        return simulate(
            inputs.open_df[stock_list], inputs.close_df[stock_list], inputs.dividend_df[stock_list], weight_df,
            start_date=ndx_vxn.TRADING_START_STR, capital_float=capital_float, share_unit_mode_str="historical",
            unadjusted_close_df=inputs.raw_close_df[stock_list], cost_model=cost_model,
        )

    return FamilyRunner(
        name_str=name_str, family_id_str="equity_cross_sectional_momentum", param_grid_dict=NDX_GRID_DICT,
        live_config_dict={"roc_month_int": 12, "stock_sma_int": 100, "top_count_int": 10}, simulate_fn=simulate_fn,
    )


def ndx_vxn_family(inputs=None) -> FamilyRunner:
    return _ndx_family("ndx_vxn", "NDX VXN", inputs)


def ndx_atr_family(inputs=None) -> FamilyRunner:
    return _ndx_family("ndx_atr", "NDX ATR", inputs)


def ndx_natr20_family(inputs=None) -> FamilyRunner:
    return _ndx_family("ndx_natr20", "NDX NATR20", inputs)


def ndx_natr20_vxn_family(inputs=None) -> FamilyRunner:
    return _ndx_family("ndx_natr20_vxn", "NDX NATR20 VXN", inputs)


def grid_return_df(family: FamilyRunner, cost_model: CostModel = DEFAULT_COST_MODEL, capital_float: float = 100_000.0) -> pd.DataFrame:
    """Daily net returns of every grid configuration (columns in grid order)."""
    return pd.DataFrame(
        {family.label_str(config_dict): family.run_config(config_dict, cost_model, capital_float).daily_return_ser for config_dict in family.config_list()}
    )
