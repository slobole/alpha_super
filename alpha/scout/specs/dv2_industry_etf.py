"""Scout spec of the DV2 industry-ETF pod (RESEARCH, Bench WIRED display; `strategies.dv2.strategy_mr_dv2_industry_etf:
DVO2IndustryEtfStrategy`): the WIRED DV2 rules on a fixed list of 19 liquid US industry ETFs.

The trade rules, sizing and execution are the DV2 spec's (alpha/scout/specs/dv2.py: features, exits, slot rule, the
`dv2_decision_fn` hook on the weights engine with `hold_nan_bool`, whole shares in adjusted units); this module adds only
the ETF eligibility. Mapped on 2026-10-02 against 64fb96e: strategies/dv2/strategy_mr_dv2_industry_etf.py (lines "I:"),
its parent strategies/dv2/strategy_mr_dv2_liquidity_floor.py (lines "F:"), alpha/engine as in dv2.py. No engine change.

Data (I: _load :104-106)
    prices          load_raw_prices(the 19 ETFs, ["$SPX"], 2009-01-01 .. today): CAPITALSPECIAL Open, High, Low, Close,
                    Volume, Turnover, Unadjusted Close, Dividend (float32); $SPX (read as $SPXTR) only widens the index.
                    Norgate's SMH and OIH start 2011-12-21 (the VanEck relaunch; the HOLDRS history is not linked).
    universe_df     `build_history_universe_df` (I :63-68): 1 once the ETF has 252 non-NaN closes counted from the
                    2009-01-02 load start through T (cumulative, causal). Every other ETF is in the data from that
                    start, so the count binds only SMH and OIH (eligible from 2012-12-21) inside the 2012+ backtest.
    Eligible ETFs (Norgate 2026-10-01): 6-10 in 2012, 9-12 in 2013-14, 11-14 in 2015-17, 13-17 since 2018; XPH is
    never eligible (ADV under $50M), XSD only from 2026-06, IGV from 2018-06, IHI from 2018-12, ITA from 2017-10.

Features at the close of T, per ETF (F: compute_signals :99-148): the four DV2 features exactly as dv2.py (float32
momentum), plus
    adv_63          mean of native Norgate Turnover over [T-62, T] (pandas rolling(63, min_periods=63)); a Turnover cell
                    that is NaN, infinite or <= 0 is NaN, so it voids every window that contains it
    raw_price       Unadjusted Close where finite and > 0, else NaN (only its NaN matters here: the ETF rule has no $5
                    bar)

Decision after the close of T (F: iterate :150-173 = DV2's; I: get_opportunities :74-85)
    candidates      `close.unstack()` minus `$` columns; members of the last universe row dated <= T; then `.dropna()`
                    (every raw field, the four DV2 features, adv_63 and raw_price non-NaN); then adv_63 > $50M; then
                    DV2 < 10, Close > SMA200, p126d_return > 0.05; sorted by NATR descending (pandas `sort_values` on
                    the eligible rows only, symbol-sorted: the spec makes the same call on the same rows by giving
                    `dv2_decision_fn` qualify = entry rule AND eligible)
    the rest        exits Close_T > High_(T-1), 10 slots of V_T / 10 in whole shares, held ETFs skipped without a slot,
                    fills Open_(T+1) x (1 +- 2.5 bp), fee max(1, 0.005 x shares), dividends net of 25% withholding:
                    exactly dv2.py (I: _new_strategy_obj :88-101)
Calendar            first decision = the close of the session before 2012-01-03 (I: run_variant :161)

    eligible_i,T = member_i,T * 1[raw_price_i,T finite] * 1[adv_63_i,T finite] * 1[adv_63_i,T > 50,000,000]

*** CRITICAL*** every feature and the eligibility at T read bars up to T only; fills at Open(T+1).

Family parameters: `Dv2Config` from dv2.py (the default = the engine configuration); the ADV floor and the history rule
are the universe's identity and stay fixed (`MIN_ADV_DOLLAR_FLOAT`, `MIN_HISTORY_SESSION_INT`).

MCPT ("spec" kind, `mcpt_option_dict`): `mcpt_matrix` = sector_ibs.bar_matrix: the 19 ETFs' TOTALRETURN daily returns
(the SD baseline), then their bars in logs (gap, high, low, close) from the session after the last first bar
(2011-12-22: SMH and OIH) to the seal. A date shuffle moves whole bars across ETFs together. Eligibility (history,
Turnover ADV, raw price) stays on its REAL dates (A8: membership and Turnover are not permuted), passed as
`eligible_mat` by `mcpt_fast_kwarg_dict`. `fast_daily_list` rebuilds OHLC and runs dv2.fast_daily_list_panel on it: every
feature recomputed (float64), the slot rule on a gross fractional-share ledger (no costs, no dividends). The replica's
features warm up from the matrix's first row, so the S5 score's 260-row warm-up starts the comparison in 2013-01.
Checked 2026-10-02 against the parity family (gross, all 27 configurations, 2013-01-08 to 2022-12-30): daily correlation
min 0.9987, median 0.9998 (net of engine costs: min 0.9984); Spearman of configuration Sharpes 0.988; the replica
Sharpe sits up to about 0.03 lower (no dividends). From 2012-01-03 the correlation is min 0.981 (the replica cannot trade
until its features warm up, about 2012-10). About 0.02 s per draw for the grid once compiled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate
from alpha.scout.specs import dv2, sector_ibs

STRATEGY_IMPORT_STR = "strategies.dv2.strategy_mr_dv2_industry_etf"
ETF_TUPLE = (  # I :53-56, in the engine's order (the spec's columns are sorted)
    "XBI", "IBB", "SMH", "SOXX", "KRE", "KBE", "XHB", "ITB", "XRT", "XOP",
    "OIH", "XME", "GDX", "IYT", "IGV", "ITA", "IHI", "XSD", "XPH",
)
BENCHMARK_STR = "$SPX"
HISTORY_START_STR = "2009-01-01"  # I :60
BACKTEST_START_STR = "2012-01-03"  # I :59
MIN_HISTORY_SESSION_INT = 252  # I :57
ADV_WINDOW_INT = 63  # F :59
MIN_ADV_DOLLAR_FLOAT = 50_000_000.0  # I :58
ENGINE_COST_MODEL = CostModel(slippage_float=0.00025, fee_per_share_float=0.005, min_fee_float=1.0)  # I :91-96
SEAL_END_STR = "2022-12-30"
LIVE_CONFIG = dv2.LIVE_CONFIG  # the WIRED DV2 rule unchanged (I: get_opportunities filters, max_positions = 10)


@dataclass(frozen=True)
class Dv2EtfInputs:
    """The DV2 spec's inputs (member_df = the history rule as of T) plus the ETF eligibility fields."""

    base: dv2.Dv2Inputs
    adv_df: pd.DataFrame  # ADV63 of native Turnover (float64, NaN unless 63 valid cells)
    raw_price_ok_df: pd.DataFrame  # bool: Unadjusted Close finite and > 0
    total_return_close_df: pd.DataFrame | None = None  # TOTALRETURN closes (MCPT baseline only)


def history_universe_df(close_df: pd.DataFrame, min_history_int: int = MIN_HISTORY_SESSION_INT) -> pd.DataFrame:
    """1 once an ETF has `min_history_int` non-NaN closes through T (I :63-68). *** CRITICAL*** cumulative to T only."""
    return (close_df.notna().cumsum() >= min_history_int).astype(int)


def adv_df_from_turnover(turnover_df: pd.DataFrame, window_int: int = ADV_WINDOW_INT) -> pd.DataFrame:
    """ADV_T = mean of native Turnover over [T - window + 1, T] (F :126-144), per column with the engine's own pandas calls
    (the frame keeps the loader's dtype, as `pd.to_numeric` does)."""
    column_dict = {}
    for symbol_str in turnover_df.columns:
        dollar_ser = pd.to_numeric(turnover_df[symbol_str], errors="coerce")
        dollar_ser = dollar_ser.where(np.isfinite(dollar_ser) & dollar_ser.gt(0.0))
        column_dict[symbol_str] = dollar_ser.rolling(window_int, min_periods=window_int).mean()
    return pd.DataFrame(column_dict, index=turnover_df.index)


def inputs_from_frames(pricing_df: pd.DataFrame, total_return_close_df: pd.DataFrame | None = None,
                       backtest_start_str: str = BACKTEST_START_STR) -> Dv2EtfInputs:
    """Inputs from a (symbol, field) price frame in the load_raw_prices layout (incl. a `$` benchmark)."""
    pricing_df = pricing_df.sort_index()
    stock_list = sorted({str(s) for s, _ in pricing_df.columns if not str(s).startswith("$")})

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: pricing_df[(s, field_str)] for s in stock_list}, index=pricing_df.index)

    universe_df = history_universe_df(field_df("Close"))
    base = dv2.inputs_from_frames(pricing_df, universe_df, backtest_start_str=backtest_start_str)
    raw_close_df = field_df("Unadjusted Close").apply(pd.to_numeric, errors="coerce")
    return Dv2EtfInputs(
        base=base, adv_df=adv_df_from_turnover(field_df("Turnover")).reindex(base.close_df.index),
        raw_price_ok_df=(np.isfinite(raw_close_df) & raw_close_df.gt(0.0)).reindex(base.close_df.index, fill_value=False),
        total_return_close_df=None if total_return_close_df is None else total_return_close_df.reindex(base.close_df.index)[stock_list],
    )


def load_inputs(end_date_str: str | None = None, total_return_bool: bool = True) -> Dv2EtfInputs:
    from data.norgate_loader import load_price_timeseries, load_raw_prices

    pricing_df = load_raw_prices(list(ETF_TUPLE), [BENCHMARK_STR], start_date=HISTORY_START_STR, end_date=end_date_str)  # I :104-106
    total_return_df = None
    if total_return_bool:
        total_return_df = pd.DataFrame({
            s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=HISTORY_START_STR, end_date_str=end_date_str)["Close"]
            for s in ETF_TUPLE
        }).astype(float)
    return inputs_from_frames(pricing_df, total_return_df)


def eligible_mat(inputs: Dv2EtfInputs, min_adv_float: float | None = None) -> np.ndarray:
    """eligible at T: history member, raw price and ADV63 present (the `dropna`), ADV63 > the floor (I :78-79;
    None = MIN_ADV_DOLLAR_FLOAT)."""
    min_adv_float = MIN_ADV_DOLLAR_FLOAT if min_adv_float is None else min_adv_float
    adv_mat = inputs.adv_df.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return inputs.base.member_df.to_numpy() & inputs.raw_price_ok_df.to_numpy() & ~np.isnan(adv_mat) & (adv_mat > min_adv_float)


def signal_mats(inputs: Dv2EtfInputs, config: dv2.Dv2Config = LIVE_CONFIG) -> dict:
    """dv2.signal_mats with qualify restricted to eligible ETFs, so the NATR sort runs on the engine's rows."""
    mats = dv2.signal_mats(inputs.base, config)
    eligible = eligible_mat(inputs)
    return {**mats, "qualify": mats["qualify"] & eligible, "member": eligible}


def simulate_config(inputs: Dv2EtfInputs, config: dv2.Dv2Config = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    base = inputs.base
    mats = signal_mats(inputs, config)
    empty_df = pd.DataFrame(columns=list(base.close_df.columns), dtype=float)
    return simulate(
        base.open_df, base.close_df, base.dividend_df, empty_df, start_date=base.backtest_start_str, capital_float=capital_float,
        share_unit_mode_str="adjusted", cost_model=cost_model, hold_nan_bool=True,
        decision_fn=dv2.dv2_decision_fn(mats["qualify"], mats["member"], mats["exit_signal"], mats["natr"], config.max_positions_int),
    )


# ---------------------------------------------------------------- MCPT replica (S5, "spec" kind)
def _sector_inputs(inputs: Dv2EtfInputs) -> sector_ibs.SectorEtfInputs:
    """The ETF bars in float64 in the sector-ETF layout that sector_ibs.bar_matrix reads."""
    if inputs.total_return_close_df is None:
        raise ValueError("The MCPT matrix needs TOTALRETURN closes: load_inputs(total_return_bool=True).")
    base = inputs.base
    return sector_ibs.SectorEtfInputs(
        open_df=base.open_df.astype(float), high_df=base.high_df.astype(float), low_df=base.low_df.astype(float),
        close_df=base.close_df.astype(float), dividend_df=base.dividend_df.astype(float).fillna(0.0),
        total_return_close_df=inputs.total_return_close_df.astype(float), backtest_start_str=base.backtest_start_str,
    )


def mcpt_matrix(inputs: Dv2EtfInputs, end_date_str: str = SEAL_END_STR) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, [TR returns of the ETFs | gap | high | low | close log bars]) from the session after the last first
    bar (SMH, OIH: 2011-12-22) to `end_date_str`; nothing is zero-filled at a listing."""
    return sector_ibs.bar_matrix(_sector_inputs(inputs), end_date_str)


def mcpt_fast_kwarg_dict(inputs: Dv2EtfInputs, date_index: pd.DatetimeIndex) -> dict:
    """Eligibility on the matrix's real dates (A8: history, Turnover and price stay put under the shuffle)."""
    eligible_df = pd.DataFrame(eligible_mat(inputs), index=inputs.base.close_df.index, columns=inputs.base.close_df.columns)
    return {"eligible_mat": eligible_df.reindex(date_index, fill_value=False).to_numpy(dtype=bool)}


def mcpt_option_dict() -> dict:
    """`PodPlan.option_dict["spec"]` for alpha.scout.reaudit.spec_mcpt."""
    return {"module_str": __name__, "matrix_fn": lambda module, inputs: module.mcpt_matrix(inputs), "asset_count_int": len(ETF_TUPLE),
            "fast_kwarg_fn": lambda module, inputs, date_index: module.mcpt_fast_kwarg_dict(inputs, date_index)}


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict], eligible_mat: np.ndarray,
                    base_config: dv2.Dv2Config = LIVE_CONFIG) -> list[np.ndarray]:
    """Gross daily returns per configuration from the (possibly date-shuffled) matrix and the real-date eligibility:
    OHLC rebuilt from the log bars, then dv2.fast_daily_list_panel with membership = eligibility. A candidate at T needs
    a complete bar, every feature, the entry rule and eligibility at T; the ledger is the DV2 replica's."""
    from alpha.scout.panel import Panel

    asset_count_int = eligible_mat.shape[1]
    column_list = [f"E{i:02d}" for i in range(asset_count_int)]
    ohlc_dict = sector_ibs.rebuilt_ohlc_dict(matrix, asset_count_int, date_index, column_list)
    member_df = pd.DataFrame(eligible_mat.astype(np.int8), index=date_index, columns=column_list)
    panel = Panel(name_str="DV2 industry ETF rebuilt bars", field_dict=ohlc_dict, member_df=member_df, snapshot_id_str="mcpt", sealed_bool=False)
    daily_list, _ = dv2.fast_daily_list_panel(panel, config_list, base_config=base_config)
    return daily_list


# ---------------------------------------------------------------- S3 (class E: the entry event against the eligible ETFs)
S3_HORIZON_TUPLE = sector_ibs.S3_HORIZON_TUPLE


def s3_panel(inputs: Dv2EtfInputs, end_date_str: str = SEAL_END_STR):
    """The 19 ETFs as an S3 panel: CAPITALSPECIAL bars, member = eligible at T with a complete bar, cut at the seal."""
    from alpha.scout.panel import Panel

    base = inputs.base
    member_df = pd.DataFrame(eligible_mat(inputs) & base.complete_df.to_numpy(), index=base.close_df.index,
                             columns=base.close_df.columns).loc[:end_date_str].astype(int)
    field_dict = {name_str: frame.astype(float).loc[:end_date_str] for name_str, frame in
                  (("Open", base.open_df), ("High", base.high_df), ("Low", base.low_df), ("Close", base.close_df))}
    return Panel(name_str="DV2 industry ETFs (eligible)", field_dict=field_dict, member_df=member_df,
                 snapshot_id_str="dv2_industry_etf_" + str(pd.Timestamp(member_df.index[-1]).date()), sealed_bool=True)


def s3_inputs(inputs: Dv2EtfInputs | None = None, end_date_str: str = SEAL_END_STR) -> dict:
    """Inputs for alpha.scout.stations.s3_edge.run_s3: event = the raw entry rule at T (DV2 < 10, Close > SMA200,
    126-session return > 5%, before the slot cap) among eligible ETFs; regime = every eligible ETF that day (excess is
    measured against the same-date eligible mean, so a market or sector-wide bounce does not count); indicator = -DV2
    (higher = more oversold); horizon = the S3 horizon nearest the live run's median holding period (sessions, entry
    open to exit open; a tie goes to the shorter horizon)."""
    inputs = inputs or load_inputs(total_return_bool=False)
    panel = s3_panel(inputs, end_date_str)
    mats = signal_mats(inputs, LIVE_CONFIG)
    frame = lambda mat: pd.DataFrame(mat, index=inputs.base.close_df.index, columns=inputs.base.close_df.columns).loc[:end_date_str]
    features = dv2._features(inputs.base, LIVE_CONFIG)
    hold_int = sector_ibs.median_holding_sessions_int(simulate_config(inputs))
    horizon_int = min(S3_HORIZON_TUPLE, key=lambda h: (abs(h - hold_int), h))
    return {"name_str": "DV2 oversold (19 industry ETFs, ADV > $50M)", "panel": panel, "regime_mask_df": panel.member_df == 1,
            "event_mask_df": frame(mats["qualify"]), "horizon_int": horizon_int, "indicator_df": frame(-features["dv2_mat"]),
            "expected_sign_int": 1}


def s3_result(input_dict: dict) -> dict:
    """Run S3 (class E) on `s3_inputs`, shaped for `PodPlan.s3_fn` and the card (alpha/scout/specs/sector_ibs.py)."""
    return sector_ibs.s3_result(input_dict)
