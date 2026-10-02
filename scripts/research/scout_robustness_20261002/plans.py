"""Registered plans for the A15 robustness diagnostics (alpha/scout/stations/robustness.py), fixed before any run.

Five pods, chosen by the owner on 2026-10-02: the two LIVE pods (TAA 3x, NDX VXN), the NATR20 sibling of NDX VXN
(A10: the live dollar-ATR score ranks partly by share price; NATR20 is the shadow candidate), and the two defensive
candidates (CORE5, and BTAL_QQQ = the WIRED TAA linearity 1/N QQQ pod). The contribution and timing diagnostics
also run, live configuration only, on every other re-audited family.

Per pod:
- eval_start_str: the evaluation window starts after the longest warm-up any draw of the random box can need, and
  ends at the vault seal (2022-12-30), so the live rule, every ablation and every draw are judged on the same days.
  Review fix (2026-10-02, before the full run completed): the first windows (TAA 2013, NDX 2002, CORE5 2009) started
  later than any warm-up needs and cut 2000-01 and 2008 out; they now start at the true longest warm-up:
  TAA and BTAL_QQQ 2012-11-01 (12 month-ends / 252 sessions after BTAL's 2011-09-13 start), NDX 2000-09-01 (ROC 18
  month-ends after the 1999-01 history start: first execution 2000-08-01), CORE5 2008-04-01 (a 252-observation
  percentile on UUP, listed 2007-02-20).
- ablation_list: components in the order they are switched off cumulatively, overlays and details first, the core
  last (decided from the rule's structure, not from any result).
- sample_fn: the random box, wider than the S4 grid, drawn with numpy's default_rng(seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha.scout.stations.robustness import AblationStep

DRAW_COUNT_INT, SEED_INT = 200, 20261002


def _log_int(rng, low_int: int, high_int: int) -> int:
    return int(round(np.exp(rng.uniform(np.log(low_int), np.log(high_int)))))


def _subset(rng, value_tuple: tuple, max_size_int: int) -> tuple:
    size_int = int(rng.integers(1, max_size_int + 1))
    return tuple(sorted(rng.choice(value_tuple, size=size_int, replace=False).tolist()))


def taa_momentum_sample(rng) -> dict:
    """momentum months: 1-4 of {1, 2, 3, 4, 6, 9, 12}; VIX-gate realised-volatility window 5-126 sessions (log-uniform)."""
    return {"momentum_month_tuple": _subset(rng, (1, 2, 3, 4, 6, 9, 12), 4), "realized_vol_window_int": _log_int(rng, 5, 126)}


def taa_linearity_sample(rng) -> dict:
    """linearity lookbacks: 1-4 of {21, 42, 63, 126, 189, 252} sessions; threshold uniform -0.0002 to +0.0002; VIX-gate
    realised-volatility window 5-126 sessions (log-uniform)."""
    return {"linearity_day_tuple": _subset(rng, (21, 42, 63, 126, 189, 252), 4),
            "linearity_threshold_float": float(rng.uniform(-0.0002, 0.0002)), "realized_vol_window_int": _log_int(rng, 5, 126)}


def ndx_sample(rng) -> dict:
    """ROC 3-18 month-ends; top 5-25 stocks; stock SMA 20-250 (log-uniform); ATR window 10-63; SPY regime SMA 50-250;
    VXN reference 16-30 (uniform)."""
    return {"roc_month_int": int(rng.integers(3, 19)), "top_count_int": int(rng.integers(5, 26)),
            "stock_sma_int": _log_int(rng, 20, 250), "atr_window_int": int(rng.integers(10, 64)),
            "regime_sma_int": int(rng.integers(50, 251)), "vxn_reference_float": float(rng.uniform(16.0, 30.0))}


def core5_sample(rng) -> dict:
    """percentile window 63-252; power 0.5-4; fast EMA 20-100; slow EMA 120-300; price filter 3-21; DBC volatility window
    21-126; DBC short volatility target 1-5%; short cap 5-20% (all uniform)."""
    return {"percentile_lookback_int": int(rng.integers(63, 253)), "percentile_power_float": float(rng.uniform(0.5, 4.0)),
            "fast_lookback_int": int(rng.integers(20, 101)), "slow_lookback_int": int(rng.integers(120, 301)),
            "price_filter_lookback_int": int(rng.integers(3, 22)), "commodity_vol_lookback_int": int(rng.integers(21, 127)),
            "commodity_short_vol_target_float": float(rng.uniform(0.01, 0.05)), "commodity_short_cap_float": float(rng.uniform(0.05, 0.20))}


@dataclass(frozen=True)
class RobustPlan:
    name_str: str  # the re-audition pod name (its bundle directory is the name with "_" for " " and "-" for "/")
    kind_str: str  # "taa", "ndx" or "core5": how the runner builds the pod
    variant_str: str
    eval_start_str: str
    ablation_list: tuple = field(default_factory=tuple)
    sample_fn: object = None


TAA_3X_ABLATION = (
    AblationStep("rank slot weights", {"slot_weight_str": "equal"}, "5/4/3/2/1 of 15 -> 1/5 per slot"),
    AblationStep("DTB3 cash hurdle", {"cash_hurdle_bool": False}, "a defensive asset qualifies on a positive score"),
    AblationStep("VIX cash gate", {"vix_gate_bool": False}, "the fallback is never sent to cash"),
    AblationStep("defensive assets", {"defensive_hold_str": "cash"}, "a qualifying slot is held in cash"),
    # Leverage is sizing, not a rule: Sharpe barely sees it, so it is a single drop (read its CAGR and Max DD).
    AblationStep("3x fallback leverage", {"fallback_str": "QQQ"}, "TQQQ -> QQQ (sizing: read CAGR and Max DD)", cumulative_bool=False),
    AblationStep("fallback", {"fallback_hold_str": "cash"}, "a failed slot is held in cash (single drop only)", cumulative_bool=False),
)
BTAL_QQQ_ABLATION = (
    AblationStep("VIX cash gate", {"vix_gate_bool": False}, "the fallback is never sent to cash"),
    AblationStep("BTAL", {"cash_asset_tuple": ("BTAL",)}, "BTAL's qualifying slot held in cash (others unchanged)"),
    AblationStep("defensive assets", {"defensive_hold_str": "cash"}, "a qualifying slot is held in cash"),
    AblationStep("fallback", {"fallback_hold_str": "cash"}, "a failed slot is held in cash (single drop only)", cumulative_bool=False),
)


def _ndx_ablation(score_name_str: str) -> tuple:
    return (
        AblationStep("VXN scaling", {"vxn_scaled_bool": False}, "slots of 1/10, no volatility scaling"),
        AblationStep("SPY regime filter", {"regime_filter_bool": False}, "always invested"),
        AblationStep("stock trend filter", {"stock_trend_filter_bool": False}, "no Close > SMA100 requirement"),
        AblationStep(f"{score_name_str} normalisation", {"atr_unit_str": "none"}, "rank on 12-month ROC alone"),
    )


CORE5_ABLATION = (
    AblationStep("DBC short", {"commodity_short_cap_float": 0.0}, "no volatility-scaled commodity short"),
    AblationStep("adaptive speed", {"adaptive_speed_bool": False}, "one EMA at the live rule's average speed (w = 1/3)"),
    AblationStep("price smoothing", {"price_filter_lookback_int": 1}, "the close itself instead of SMA10"),
    AblationStep("trend rule", {"trend_rule_bool": False}, "static 20% sleeves, month-end rebalancing"),
)

PLAN_DICT = {
    "taa_3x": RobustPlan("TAA 3x", "taa", "taa_3x", "2012-11-01", TAA_3X_ABLATION, taa_momentum_sample),
    "btal_qqq": RobustPlan("TAA linearity 1/N QQQ", "taa", "taa_lin_1n_qqq", "2012-11-01", BTAL_QQQ_ABLATION, taa_linearity_sample),
    "ndx_vxn": RobustPlan("NDX VXN", "ndx", "ndx_vxn", "2000-09-01", _ndx_ablation("dollar ATR"), ndx_sample),
    "ndx_natr20_vxn": RobustPlan("NDX NATR20 VXN", "ndx", "ndx_natr20_vxn", "2000-09-01", _ndx_ablation("NATR20"), ndx_sample),
    "core5": RobustPlan("CORE5", "core5", "core5", "2008-04-01", CORE5_ABLATION, core5_sample),
}
