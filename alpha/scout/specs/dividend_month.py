"""Scout spec: the dividend month premium (Hartzmark and Solomon 2013), the first P8 discovery family (design A20).

Registered 2026-10-16 before any result (ledger row 52, `dividend_month_premium_20261016`, family
dividend_seeking_demand). There is no engine strategy: a discovery family reaches the real engine only after S7.

Rule, decided at the close T of the last session of month m-1 (or `decision_offset_int` sessions earlier, the S4 luck
band), from data dated <= T:

    eligible_i(T)  = member_i(T) and >= 252 finite closes up to T and UnadjClose_i(T) >= $5 and Turnover_i(T) > 0
    ex_i(p)        = an ex-dividend date in calendar month p. The panel books a dividend on the LAST CUM-DIVIDEND
                     session (the engine credits it to the position held at that close, before the next open), so the
                     ex-date is the next session: `ex_dividend_df` shifts the column by one session
    m              = month(T) + 1;  m-3 and m-12 are always before month(T), so they are known at T
    predicted_i    = "m3": ex_i(m-3) | "m12": ex_i(m-12) | "m3_or_m12": either | "m3_and_m12": both
    long weights   = predicted and eligible: 1/n (EW), or last dividend / Close_T normalised (yield)
    short weights  = eligible, not predicted: -1/k (the pod is dollar-neutral, gross 2.0)

Executed at the open of the next session (weights engine, historical share units, 25% withholding on long dividends,
1%/yr borrow on shorts, fund-scale capital $10M) and held to the next rebalance.

The S3 study and the S5 replica read TOTAL returns (Close_t + Dividend_t) / Close_(t-1): on price-only returns a payer
loses its dividend on the ex-date, and that drop is the same size as the premium being tested.

S5 null (registered): the per-asset permutation of A8 applied to the RETURNS only. The dividend calendar, the
eligibility and the weights come from the real panel and stay fixed (`signal_dict`), so the null breaks the link
between the predicted month and the returns without destroying the calendar that defines the signal. This null has
no calibration run of its own (A8 calibrated reversal and momentum); the card says so.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import BorrowModel, CostModel, WeightsResult, simulate
from alpha.scout.searches import _hold_daily

NAME_STR = "Dividend month premium"
FAMILY_ID_STR = "dividend_seeking_demand"
REGISTRATION_ID_STR = "dividend_month_premium_20261016"
PANEL_NAME_STR, REPLICATION_PANEL_NAME_STR = "S&P 500", "Nasdaq 100"
LOOKBACK_RULE_TUPLE = ("m3", "m12", "m3_or_m12", "m3_and_m12")
WEIGHTING_TUPLE = ("EW", "yield")
GRID_DICT = {"lookback_rule_str": LOOKBACK_RULE_TUPLE, "weighting_str": WEIGHTING_TUPLE}
FUND_CAPITAL_FLOAT = 10_000_000.0
ENGINE_COST_MODEL = CostModel()
BORROW_MODEL = BorrowModel()
S3_HORIZON_INT = 21  # one month of sessions from the entry open
YIELD_STALE_SESSION_INT = 400  # a dividend older than this is not a predicted yield


@dataclass(frozen=True)
class DividendMonthConfig:
    lookback_rule_str: str = "m3_or_m12"
    weighting_str: str = "EW"
    decision_offset_int: int = 0
    min_history_int: int = 252
    min_price_float: float = 5.0

    def __post_init__(self) -> None:
        if self.lookback_rule_str not in LOOKBACK_RULE_TUPLE:
            raise ValueError(f"lookback_rule_str must be one of {LOOKBACK_RULE_TUPLE}.")
        if self.weighting_str not in WEIGHTING_TUPLE:
            raise ValueError(f"weighting_str must be one of {WEIGHTING_TUPLE}.")
        if self.decision_offset_int < 0:
            raise ValueError("decision_offset_int must be non-negative.")


LIVE_CONFIG = DividendMonthConfig()  # the registered primary configuration


@dataclass
class DividendMonthInputs:
    panel: object  # alpha.scout.panel.Panel
    cache_dict: dict = field(default_factory=dict)


def load_inputs(panel_name_str: str = PANEL_NAME_STR, vault_opening_row_id_int: int | None = None, ledger=None) -> DividendMonthInputs:
    """The sealed panel by default; the full panel only with this family's verified vault-opening row (S7)."""
    from alpha.scout.panel import load_panel

    panel = load_panel(panel_name_str, vault_opening_row_id_int=vault_opening_row_id_int,
                       family_id_str=FAMILY_ID_STR if vault_opening_row_id_int is not None else None, ledger=ledger)
    return DividendMonthInputs(panel=panel)


# ---------------------------------------------------------------- signal
def decision_row_vec(date_index: pd.DatetimeIndex, decision_offset_int: int = 0) -> np.ndarray:
    from alpha.scout.specs.core5 import month_end_flag_vec

    row_vec = np.flatnonzero(month_end_flag_vec(date_index, decision_offset_int))
    return row_vec[row_vec + 1 < len(date_index)]  # a decision needs a next session to execute


def eligible_mat(panel, config: DividendMonthConfig = LIVE_CONFIG) -> np.ndarray:
    close_df = panel.field("Close")
    finite_df = close_df.notna()
    with np.errstate(invalid="ignore"):
        return ((panel.member_df == 1).to_numpy() & finite_df.to_numpy()
                & (finite_df.cumsum() >= config.min_history_int).to_numpy()
                & (panel.field("Unadjusted Close") >= config.min_price_float).to_numpy()
                & (panel.field("Turnover") > 0).to_numpy())


def ex_dividend_df(panel) -> pd.DataFrame:
    """Dividend per share on its ex-date (the panel's Dividend column shifted forward one session, see the docstring)."""
    return panel.field("Dividend").fillna(0.0).shift(1).fillna(0.0)


def ex_month_df(panel) -> pd.DataFrame:
    """True where a symbol had an ex-dividend date in that calendar month (PeriodIndex rows)."""
    paid_df = ex_dividend_df(panel) > 0
    return paid_df.groupby(panel.date_index.to_period("M")).any()


def predicted_mat(panel, row_vec: np.ndarray, lookback_rule_str: str, ex_df: pd.DataFrame | None = None) -> np.ndarray:
    ex_df = ex_month_df(panel) if ex_df is None else ex_df
    target_period_index = panel.date_index[row_vec].to_period("M") + 1

    def ex_at(lag_int: int) -> np.ndarray:
        return ex_df.reindex(target_period_index - lag_int, fill_value=False).to_numpy(dtype=bool)

    if lookback_rule_str == "m3":
        return ex_at(3)
    if lookback_rule_str == "m12":
        return ex_at(12)
    if lookback_rule_str == "m3_or_m12":
        return ex_at(3) | ex_at(12)
    return ex_at(3) & ex_at(12)


def predicted_yield_mat(panel, row_vec: np.ndarray) -> np.ndarray:
    dividend_df = panel.field("Dividend")
    last_dividend_df = dividend_df.where(dividend_df > 0).ffill(limit=YIELD_STALE_SESSION_INT)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (last_dividend_df.to_numpy(dtype=float) / panel.field("Close").to_numpy(dtype=float))[row_vec]


def signal_for(panel, config: DividendMonthConfig, cache_dict: dict | None = None) -> dict:
    """Decision rows, long weights, short weights and the equal-weight eligible baseline weights at each decision."""
    cache_dict = {} if cache_dict is None else cache_dict
    key_tuple = (config.lookback_rule_str, config.weighting_str, config.decision_offset_int, config.min_history_int, config.min_price_float)
    if key_tuple in cache_dict:
        return cache_dict[key_tuple]
    base_key_tuple = ("base", config.decision_offset_int, config.min_history_int, config.min_price_float)
    if base_key_tuple not in cache_dict:
        row_vec = decision_row_vec(panel.date_index, config.decision_offset_int)
        cache_dict[base_key_tuple] = (row_vec, eligible_mat(panel, config)[row_vec])
    if "ex_df" not in cache_dict:
        cache_dict["ex_df"] = ex_month_df(panel)
    row_vec, eligible_at_mat = cache_dict[base_key_tuple]
    payer_mat = predicted_mat(panel, row_vec, config.lookback_rule_str, cache_dict["ex_df"]) & eligible_at_mat
    if config.weighting_str == "EW":
        long_mat = payer_mat.astype(float)
    else:
        yield_key_tuple = ("yield", config.decision_offset_int)
        if yield_key_tuple not in cache_dict:
            cache_dict[yield_key_tuple] = predicted_yield_mat(panel, row_vec)
        yield_mat = np.nan_to_num(cache_dict[yield_key_tuple], nan=0.0, posinf=0.0, neginf=0.0)
        long_mat = np.where(payer_mat, np.clip(yield_mat, 0.0, None), 0.0)
    long_mat = long_mat / np.maximum(long_mat.sum(axis=1, keepdims=True), 1e-12)
    short_mat = (eligible_at_mat & ~payer_mat).astype(float)
    short_mat = -short_mat / np.maximum(short_mat.sum(axis=1, keepdims=True), 1.0)
    baseline_mat = eligible_at_mat.astype(float)
    baseline_mat = baseline_mat / np.maximum(baseline_mat.sum(axis=1, keepdims=True), 1.0)
    signal_dict = {"row_vec": row_vec, "long_mat": long_mat, "short_mat": short_mat, "baseline_mat": baseline_mat,
                   "payer_count_vec": payer_mat.sum(axis=1), "eligible_count_vec": eligible_at_mat.sum(axis=1)}
    cache_dict[key_tuple] = signal_dict
    return signal_dict


# ---------------------------------------------------------------- engine run (S4, S6, S7, S8)
def rebalance_weight_df(inputs: DividendMonthInputs, config: DividendMonthConfig = LIVE_CONFIG) -> pd.DataFrame:
    panel = inputs.panel
    signal_dict = signal_for(panel, config, inputs.cache_dict)
    execution_index = panel.date_index[signal_dict["row_vec"] + 1]
    return pd.DataFrame(signal_dict["long_mat"] + signal_dict["short_mat"], index=execution_index, columns=panel.symbol_list)


def simulate_config(inputs: DividendMonthInputs, config: DividendMonthConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = FUND_CAPITAL_FLOAT) -> WeightsResult:
    panel = inputs.panel
    weight_df = rebalance_weight_df(inputs, config)
    return simulate(
        panel.field("Open"), panel.field("Close"), panel.field("Dividend").fillna(0.0), weight_df, weight_df.index[0],
        capital_float=capital_float, share_unit_mode_str="historical", unadjusted_close_df=panel.field("Unadjusted Close"),
        cost_model=cost_model, allow_short_bool=True, borrow_model=BORROW_MODEL,
    )


def family(inputs: DividendMonthInputs | None = None):
    """The registered grid as a FamilyRunner. The family's default capital is fund scale ($10M, registered); an
    explicit other capital (S4's $30K small-account run) is honoured."""
    from alpha.scout.family import FamilyRunner

    inputs = inputs or load_inputs()

    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float) -> WeightsResult:
        capital_float = FUND_CAPITAL_FLOAT if capital_float == 100_000.0 else capital_float
        return simulate_config(inputs, dataclasses.replace(LIVE_CONFIG, **config_dict), cost_model, capital_float)

    return FamilyRunner(
        name_str=NAME_STR, family_id_str=FAMILY_ID_STR, param_grid_dict=GRID_DICT,
        live_config_dict={k: getattr(LIVE_CONFIG, k) for k in GRID_DICT}, simulate_fn=simulate_fn,
    )


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    """S8 provider: the frozen configuration on the newest full panel (opened by the verified shadow_start row). The
    shadow monitor keeps only the sessions from the shadow's start date."""
    from alpha.scout.panel import load_panel

    panel = load_panel(PANEL_NAME_STR, family_id_str=FAMILY_ID_STR, shadow_start_row_id_int=shadow_row_dict["row_id_int"])
    if panel.sealed_bool:
        raise RuntimeError("The shadow_start row did not verify; the provider refuses to run on a sealed panel.")
    config = dataclasses.replace(LIVE_CONFIG, **shadow_row_dict["frozen_config_dict"])
    # *** CRITICAL*** returns before the start (vault bars used as warm-up) never leave the spec.
    return simulate_config(DividendMonthInputs(panel=panel), config).daily_return_ser.loc[shadow_row_dict["start_date_str"]:]


# ---------------------------------------------------------------- fast replica (S5 MCPT)
def total_return_mat(panel) -> np.ndarray:
    close_mat = panel.field("Close").to_numpy(dtype=float)
    dividend_mat = ex_dividend_df(panel).to_numpy(dtype=float)  # on the ex-date, when the price drops
    previous_mat = np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((close_mat + dividend_mat) / previous_mat - 1.0, nan=0.0, posinf=0.0, neginf=0.0)


def fast_daily_list_panel(panel, config_list: list[dict], signal_dict: dict | None = None):
    """(gross daily long-book total returns per configuration, equal-weight eligible baseline) on `panel`'s returns.

    `signal_dict` (label -> signal_for output, plus "baseline") holds the weights built from the REAL panel; the MCPT
    passes it so that only the returns are permuted. Without it the signal is built from `panel` itself."""
    return_mat = total_return_mat(panel)
    cache_dict: dict = {}
    daily_list = []
    baseline_signal = None
    for config_dict in config_list:
        config = dataclasses.replace(LIVE_CONFIG, **config_dict)
        label_str = f"{config.lookback_rule_str}|{config.weighting_str}|{config.decision_offset_int}"
        signal = signal_dict[label_str] if signal_dict is not None else signal_for(panel, config, cache_dict)
        daily_list.append(_hold_daily(signal["long_mat"], signal["row_vec"], return_mat))
        if baseline_signal is None:
            baseline_signal = signal
    baseline_vec = _hold_daily(baseline_signal["baseline_mat"], baseline_signal["row_vec"], return_mat)
    return daily_list, baseline_vec


def mcpt_signal_dict(panel, config_list: list[dict]) -> dict:
    cache_dict: dict = {}
    out_dict = {}
    for config_dict in config_list:
        config = dataclasses.replace(LIVE_CONFIG, **config_dict)
        signal = signal_for(panel, config, cache_dict)
        out_dict[f"{config.lookback_rule_str}|{config.weighting_str}|{config.decision_offset_int}"] = {
            k: signal[k] for k in ("row_vec", "long_mat", "baseline_mat")}
    return out_dict


# ---------------------------------------------------------------- S1 and S3
def total_return_panel(panel):
    """The panel with Open / High / Low / Close multiplied by the cumulative dividend factor, so every price ratio is a
    total return: Close'_t / Close'_(t-1) = (Close_t / Close_(t-1)) x (1 + ExDividend_t / Close_(t-1)). An entry at the
    open of an ex-date does not receive that dividend (the factor already includes it at the open)."""
    close_df = panel.field("Close")
    with np.errstate(divide="ignore", invalid="ignore"):
        step_df = (1.0 + ex_dividend_df(panel) / close_df.shift(1)).where(close_df.shift(1).notna(), 1.0).fillna(1.0)
    factor_df = step_df.astype(float).cumprod()
    field_dict = dict(panel.field_dict)
    for field_str in ("Open", "High", "Low", "Close"):
        field_dict[field_str] = panel.field(field_str).astype(float) * factor_df
    return dataclasses.replace(panel, field_dict=field_dict, snapshot_id_str=panel.snapshot_id_str + "|total_return")


def predicted_feature(lookback_rule_str: str = LIVE_CONFIG.lookback_rule_str):
    """S1 contract: at each session T, 1.0 if the symbol is predicted to pay in month(T) + 1, from ex-dates <= T."""
    from alpha.scout.features import Feature

    def compute(panel) -> pd.DataFrame:
        row_vec = np.arange(len(panel.date_index))
        return pd.DataFrame(predicted_mat(panel, row_vec, lookback_rule_str).astype(float), index=panel.date_index, columns=panel.symbol_list)

    return Feature(f"predicted_dividend_month_{lookback_rule_str}", compute, 260, "scale_invariant",
                   "1 if the symbol had an ex-dividend date in month m-3 / m-12 of the next month m (rule as named)")


def s3_inputs(panel=None, config: DividendMonthConfig = LIVE_CONFIG, panel_name_str: str = PANEL_NAME_STR) -> dict:
    """run_s3 inputs on the total-return panel: events at the decision rows only (one date per month), regime =
    eligible at those rows, horizon 21 sessions from the next open (month m), liquidity = 63-session turnover rank."""
    from alpha.scout import features as scout_features
    from alpha.scout.panel import load_panel

    panel = panel if panel is not None else load_panel(panel_name_str)
    row_vec = decision_row_vec(panel.date_index, config.decision_offset_int)
    eligible_at_mat = eligible_mat(panel, config)[row_vec]
    payer_mat = predicted_mat(panel, row_vec, config.lookback_rule_str) & eligible_at_mat
    shape_tuple = (len(panel.date_index), len(panel.symbol_list))
    regime_mat, event_mat = np.zeros(shape_tuple, dtype=bool), np.zeros(shape_tuple, dtype=bool)
    regime_mat[row_vec], event_mat[row_vec] = eligible_at_mat, payer_mat
    frame = lambda mat: pd.DataFrame(mat, index=panel.date_index, columns=panel.symbol_list)  # noqa: E731
    return {"name_str": f"Predicted dividend month ({panel_name_str} members, {config.lookback_rule_str})",
            "panel": total_return_panel(panel), "regime_mask_df": frame(regime_mat), "event_mask_df": frame(event_mat),
            "horizon_int": S3_HORIZON_INT, "indicator_df": None,
            "liquidity_rank_df": scout_features.turnover_rank(63).compute_fn(panel), "expected_sign_int": 1}
