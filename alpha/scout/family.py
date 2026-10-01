"""A strategy family as Scout runs it: a registered grid around a spec, executed by the parity weights engine.

`FamilyRunner.run_config(config_dict, cost_model, capital_float)` returns the engine result of one configuration;
the grid order is `Registration.grid_config_list` order (parameter names sorted, values in registered order), so
the plateau grid shape is `grid_shape_tuple`. `decision_offset_int` is not a grid axis: it is the luck band (S4).

The two LIVE pods are wired here (P5): TAA 3x and NDX VXN, executed exactly as the identity gate executes them.
The gated NDX siblings (plain ATR, NATR20, NATR20 VXN) reuse the NDX grid around their own engine config.
The gated TAA engine variants (1/N, linearity, 2x) are wired through `taa_variant_family` the same way.
Inflation Compass and its QQQ variant (PM_READY) are wired through `compass_family`.
CORE5 adaptive macro (PM_READY, gated 2026-10-01) is `core5_family`.
The PM_READY pods TFI and Trinity (2026-10-01) are wired through `tfi_family` and `trinity_family`.
The month-end rebalancing flow (PM_READY, MOC execution) is `eom_family` (no luck band: offset_count_int = 1).
The PM_READY sector ETF IBS event pods (2026-10-02) are `sector_ibs_family` and `dispersion_ibs_family`.
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


# ---------------------------------------------------------------- Inflation Compass
# Three axes around the engine's values, each one step either side (27 configurations):
# - growth_sma_int: the SPY trend window of the growth axis, 200 +- 50 sessions;
# - inflation_threshold_float: the T5YIE level threshold, 2.0 +- 0.2 pp (the 2026-09-28 study's one-axis check);
# - trend_lookback_int: the inflation-trend horizon, used for BOTH the T5YIE change anchor and the sector-basket slope
#   (the source rule ties them at 60 sessions, about a quarter); 40 / 60 / 80 as in that study.
# The 2026-09-28 study's 700-cell grid moved the two windows separately; tying them keeps the family to one concept per
# axis and inside the 9-36 configuration budget. decision_offset_int is the luck band (S4), not an axis.
COMPASS_GRID_DICT = {
    "growth_sma_int": (150, 200, 250),
    "inflation_threshold_float": (1.8, 2.0, 2.2),
    "trend_lookback_int": (40, 60, 80),
}
COMPASS_FAMILY_NAME_DICT = {"compass": "Inflation Compass", "compass_qqq": "Inflation Compass QQQ"}


def compass_family(variant_name_str: str, inputs=None) -> FamilyRunner:
    """A gated Compass variant (alpha/scout/specs/compass.py VARIANT_DICT) as a family. The cost model is the caller's
    (S4 passes the house parity costs); the identity gate uses the engine's own, `compass.ENGINE_COST_MODEL`."""
    from alpha.scout.specs import compass

    base_config = compass.VARIANT_DICT[variant_name_str].config
    if base_config.breakeven_lookback_int != base_config.asset_slope_lookback_int:
        raise ValueError("The Compass grid ties the two inflation-trend windows; the base config must too.")
    inputs = inputs or compass.load_inputs(config=base_config)

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        weight_df = compass.rebalance_weight_df(inputs, compass.config_from_dict(base_config, config_dict))
        return simulate(
            inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
            capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=cost_model,
        )

    return FamilyRunner(
        name_str=COMPASS_FAMILY_NAME_DICT[variant_name_str], family_id_str="macro_regime_allocation",
        param_grid_dict=COMPASS_GRID_DICT, simulate_fn=simulate_fn,
        live_config_dict={"growth_sma_int": base_config.growth_sma_int, "inflation_threshold_float": base_config.inflation_threshold_float,
                          "trend_lookback_int": base_config.breakeven_lookback_int},
    )


# ---------------------------------------------------------------- CORE5 adaptive macro
# The decision is SMA_n > AMA per sleeve, with the AMA's speed blended between EMA(fast) and EMA(slow) by the drawdown
# percentile. The three horizons ARE the trend rule, so they are the axes: the filter n (5, 10, 20 sessions), the fast
# end (25, 50, 100) and the slow end (150, 200, 300), each halving / doubling around the live (10, 50, 200) and ordered
# ascending; every slow value exceeds every fast value. 27 configurations. The adaptation (126-session percentile,
# power 2) and the DBC short sizing (2.5% / vol, cap 10%) stay at the live values: they shape how the AMA moves and
# how big the short is, not which trend horizon the rule follows.
CORE5_GRID_DICT = {
    "price_filter_lookback_int": (5, 10, 20),
    "fast_lookback_int": (25, 50, 100),
    "slow_lookback_int": (150, 200, 300),
}


def core5_family(inputs=None) -> FamilyRunner:
    """CORE5 adaptive macro (alpha/scout/specs/core5.py) as a family; the live config is the gated engine default."""
    from alpha.scout.specs import core5

    inputs = inputs or core5.load_inputs()

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        config = dataclasses.replace(core5.LIVE_CONFIG, **config_dict)
        return core5.simulate_config(inputs, config, cost_model=cost_model, capital_float=capital_float)

    return FamilyRunner(
        # Mechanically a per-asset trend rule (SMA vs adaptive MA per sleeve), not a macro regime map: its trials count
        # in the trend family (decided 2026-10-02, P6b).
        name_str="CORE5", family_id_str="time_series_trend_and_breakout", param_grid_dict=CORE5_GRID_DICT,
        live_config_dict={name_str: getattr(core5.LIVE_CONFIG, name_str) for name_str in CORE5_GRID_DICT}, simulate_fn=simulate_fn,
    )


# ---------------------------------------------------------------- TFI and Trinity (PM_READY, 2026-10-01)
# TFI: the rule asks "is the spread above its own long-run typical level?". Two axes move that question around the
# frozen rule (the default is the expanding median): how much history defines "typical" (rolling 5, 10 or 20 years,
# then the expanding history, coded 0 and placed last as the longest), and how high the bar sits (the 40th, 50th or
# 60th percentile of that history). The spreads, the sleeve weights and the publication lag are the rule's identity,
# not tuning knobs, and stay fixed. 12 configurations.
TFI_GRID_DICT = {
    "history_month_int": (60, 120, 240, 0),
    "threshold_quantile_float": (0.4, 0.5, 0.6),
}
# Trinity: the three numbers that define a volatility-targeted inverse-volatility book: the asset volatility window
# (one, three and six months), the base-portfolio volatility window of the overlay (same three) and the volatility
# target with its trigger kept 0.5 pp above it (6/6.5%, the live 8/8.5%, 10/10.5%). The 5 pp no-trade band is an
# execution detail and stays fixed. 27 configurations.
TRINITY_GRID_DICT = {
    "asset_vol_lookback_int": (21, 63, 126),
    "portfolio_vol_lookback_int": (21, 63, 126),
    "vol_target_tuple": ((0.06, 0.065), (0.08, 0.085), (0.10, 0.105)),
}


def _spec_family(spec_module_str: str, name_str: str, family_id_str: str, grid_dict: dict, inputs=None) -> FamilyRunner:
    """A spec with `load_inputs`, `LIVE_CONFIG` and `simulate_config(inputs, config, cost_model, capital)`. The cost
    model is the caller's (S4 passes CostModel(); the engines' own costs are each spec's ENGINE_COST_MODEL)."""
    import importlib

    spec_module = importlib.import_module(f"alpha.scout.specs.{spec_module_str}")
    inputs = inputs or spec_module.load_inputs()

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        config = dataclasses.replace(spec_module.LIVE_CONFIG, **config_dict)
        return spec_module.simulate_config(inputs, config, cost_model, capital_float)

    return FamilyRunner(
        name_str=name_str, family_id_str=family_id_str, param_grid_dict=grid_dict,
        live_config_dict={k: getattr(spec_module.LIVE_CONFIG, k) for k in grid_dict}, simulate_fn=simulate_fn,
    )


def tfi_family(inputs=None) -> FamilyRunner:
    return _spec_family("tfi", "TFI", "macro_regime_allocation", TFI_GRID_DICT, inputs)


def trinity_family(inputs=None) -> FamilyRunner:
    # Inverse-volatility weights under a volatility target: allocation by estimated risk (decided 2026-10-02, P6b).
    return _spec_family("trinity", "Trinity", "optimized_low_risk_allocation", TRINITY_GRID_DICT, inputs)


# ---------------------------------------------------------------- Month-end rebalancing flow (PM_READY, 2026-10-01)
# The rule bets that 60/40 rebalancers trade against the month's stock/bond drift in the last sessions and reverse
# early next month. Three axes, one step either side of the live rule (27 configurations):
# - entry_dtme_int: when the final leg enters (dtme 5, 6, 7); the measure stays one session before the entry;
# - exit_session_int: how long the early-month reversal is held (sessions 3, 5, 7 of the next month);
# - cut_tuple: how extreme the drift must be to act, as (low, high) cuts on the prior-month CDF F, from thin tails to
#   wide ones: (0.1, 0.7), the live quintile rule (0.2, 0.6), (0.3, 0.5).
# The leg weights (100% / 50-50 / -100%), the 60/40 reference and the 24-month warm-up are the rule's identity and stay
# fixed. There is no luck band (offset_count_int = 1): the decision day IS the calendar hypothesis, so moving it is a
# different rule; the two timing axes probe the day instead.
EOM_GRID_DICT = {
    "entry_dtme_int": (5, 6, 7),
    "exit_session_int": (3, 5, 7),
    "cut_tuple": ((0.1, 0.7), (0.2, 0.6), (0.3, 0.5)),
}


def eom_family(inputs=None) -> FamilyRunner:
    family = _spec_family("eom", "EOM flow", "calendar_and_flow", EOM_GRID_DICT, inputs)
    family.offset_count_int = 1
    return family


# ---------------------------------------------------------------- Sector ETF IBS event pods (PM_READY, 2026-10-02)
# Daily event rules: there is no rebalance schedule, so there is no luck band (offset_count_int = 1; the configs refuse a
# non-zero decision_offset_int). The axes are the rule's three decisions, each one step either side of the live value
# (27 configurations each):
# - entry_ibs_max_float: how deep in the day's range the close must be to buy, halved / doubled around the live value
#   (downshock 0.025 / 0.05 / 0.10; dispersion 0.05 / 0.10 / 0.20);
# - exit_ibs_min_float: how strong the rebound close must be to sell, its distance to the top of the range halved /
#   doubled (0.80 / 0.90 / 0.95);
# - downshock: max_positions_int 3 / 5 / 7 at the fixed 1.5 / 11 of AUM per entry (gross cap 41% / 68% / 95%);
#   dispersion (no slot cap, a 1/N sleeve per ETF): min_relative_range_float, the "wide day" bar of both entry and exit,
#   halved / doubled (0.5 / 1.0 / 2.0 x the trailing standard deviation of the log range).
# The windows (ATR 14, range 21), the downshock bar and the SMA200 gate are the variants' identity and stay fixed.
SECTOR_IBS_GRID_DICT = {
    "entry_ibs_max_float": (0.025, 0.05, 0.10),
    "exit_ibs_min_float": (0.80, 0.90, 0.95),
    "max_positions_int": (3, 5, 7),
}
DISPERSION_IBS_GRID_DICT = {
    "entry_ibs_max_float": (0.05, 0.10, 0.20),
    "exit_ibs_min_float": (0.80, 0.90, 0.95),
    "min_relative_range_float": (0.5, 1.0, 2.0),
}
DISPERSION_IBS_NAME_DICT = {
    "dispersion_ibs_kie_ihi_xlc": "Dispersion IBS KIE IHI XLC",
    "dispersion_ibs_kie_ihi_xlc_sma200": "Dispersion IBS KIE IHI XLC SMA200",
    "dispersion_ibs_kie_ihi_sma200": "Dispersion IBS KIE IHI SMA200",
}


def sector_ibs_family(inputs=None) -> FamilyRunner:
    """US sector ETF IBS downshock, VOX/IYR basket (alpha/scout/specs/sector_ibs.py). The cost model is the caller's; the
    identity gate uses the engine's, `sector_ibs.ENGINE_COST_MODEL` (= CostModel())."""
    family = _spec_family("sector_ibs", "Sector IBS VOX IYR", "etf_short_term_reversal", SECTOR_IBS_GRID_DICT, inputs)
    family.offset_count_int = 1  # a daily event rule: no rebalance offset
    return family


def dispersion_ibs_family(variant_name_str: str, inputs=None) -> FamilyRunner:
    """A sector-dispersion IBS variant (alpha/scout/specs/sector_dispersion_ibs.py VARIANT_DICT). The cost model is the
    caller's; the identity gate uses the pod's own, `sector_dispersion_ibs.ENGINE_COST_MODEL`."""
    from alpha.scout.specs import sector_dispersion_ibs

    base_config = sector_dispersion_ibs.VARIANT_DICT[variant_name_str].config
    inputs = inputs or sector_dispersion_ibs.load_inputs(variant_name_str)

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        return sector_dispersion_ibs.simulate_config(inputs, dataclasses.replace(base_config, **config_dict), cost_model, capital_float)

    return FamilyRunner(
        name_str=DISPERSION_IBS_NAME_DICT[variant_name_str], family_id_str="etf_short_term_reversal",
        param_grid_dict=DISPERSION_IBS_GRID_DICT, live_config_dict={k: getattr(base_config, k) for k in DISPERSION_IBS_GRID_DICT},
        simulate_fn=simulate_fn, offset_count_int=1,  # a daily event rule: no rebalance offset
    )


def grid_return_df(family: FamilyRunner, cost_model: CostModel = DEFAULT_COST_MODEL, capital_float: float = 100_000.0) -> pd.DataFrame:
    """Daily net returns of every grid configuration (columns in grid order)."""
    return pd.DataFrame(
        {family.label_str(config_dict): family.run_config(config_dict, cost_model, capital_float).daily_return_ser for config_dict in family.config_list()}
    )
