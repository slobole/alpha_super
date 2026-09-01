"""Engine-native passive BIL control for portfolio value-add research.

The control makes one predetermined investment decision from ``Close_T`` and
fills whole shares at ``Open_(T+1)``. It never rebalances or reinvests cash.
BIL execution and marks use Norgate ``CAPITALSPECIAL`` prices. The CORE5-matched
profile credits dividends with 25% withholding and pays 0% on residual cash.
The Tactical-FI-matched profile credits gross dividends and applies the same
causal DGS3MO ACT/365 cash return. Negative cash financing is never modeled.

This module is deliberately research-only. It is absent from the strategy
registry and every LIVE, broker, scheduler, allocation, and release surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    load_raw_prices,
)


STRATEGY_NAME_STR = "strategy_control_passive_bil"
ASSET_STR = "BIL"
BENCHMARK_TUPLE = ("$SPX",)
DEFAULT_SLIPPAGE_FLOAT = 0.00025
DEFAULT_COMMISSION_PER_SHARE_FLOAT = 0.005
DEFAULT_COMMISSION_MINIMUM_FLOAT = 1.0
DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT = 0.25
TACTICAL_FI_SLIPPAGE_FLOAT = 0.0005
TACTICAL_FI_COMMISSION_PER_SHARE_FLOAT = 0.0
TACTICAL_FI_COMMISSION_MINIMUM_FLOAT = 0.0
CORE5_MATCHED_PROFILE_STR = "core5_net25_zero_cash"
TACTICAL_FI_MATCHED_PROFILE_STR = "tactical_fi_gross_dgs3mo_cash"
ACCOUNTING_PROFILE_TUPLE = (
    CORE5_MATCHED_PROFILE_STR,
    TACTICAL_FI_MATCHED_PROFILE_STR,
)
CASH_INTEREST_LEDGER_COLUMN_TUPLE = (
    "date",
    "positive_cash_base_float",
    "cash_return_float",
    "cash_interest_float",
)


@dataclass(frozen=True)
class PassiveBilConfig:
    """Frozen accounting and execution contract for the passive control."""

    asset_str: str = ASSET_STR
    benchmark_tuple: tuple[str, ...] = BENCHMARK_TUPLE
    history_start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = DEFAULT_SLIPPAGE_FLOAT
    commission_per_share_float: float = DEFAULT_COMMISSION_PER_SHARE_FLOAT
    commission_minimum_float: float = DEFAULT_COMMISSION_MINIMUM_FLOAT
    dividend_withholding_rate_float: float = (
        DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT
    )
    accounting_profile_str: str = CORE5_MATCHED_PROFILE_STR

    def __post_init__(self) -> None:
        if self.asset_str != ASSET_STR:
            raise ValueError("The passive control asset must remain BIL.")
        if tuple(self.benchmark_tuple) != BENCHMARK_TUPLE:
            raise ValueError("The passive control benchmark must remain $SPX.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if min(
            self.slippage_float,
            self.commission_per_share_float,
            self.commission_minimum_float,
        ) < 0.0:
            raise ValueError("Trading costs must be non-negative.")
        if not 0.0 <= self.dividend_withholding_rate_float <= 1.0:
            raise ValueError(
                "dividend_withholding_rate_float must be between zero and one."
            )
        if self.accounting_profile_str not in ACCOUNTING_PROFILE_TUPLE:
            raise ValueError(
                f"accounting_profile_str must be one of {ACCOUNTING_PROFILE_TUPLE}."
            )
        expected_withholding_rate_float = (
            0.0
            if self.accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT
        )
        if not np.isclose(
            self.dividend_withholding_rate_float,
            expected_withholding_rate_float,
        ):
            raise ValueError(
                "Dividend withholding does not match the selected accounting profile."
            )
        expected_cost_tuple = (
            (
                TACTICAL_FI_SLIPPAGE_FLOAT,
                TACTICAL_FI_COMMISSION_PER_SHARE_FLOAT,
                TACTICAL_FI_COMMISSION_MINIMUM_FLOAT,
            )
            if self.accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else (
                DEFAULT_SLIPPAGE_FLOAT,
                DEFAULT_COMMISSION_PER_SHARE_FLOAT,
                DEFAULT_COMMISSION_MINIMUM_FLOAT,
            )
        )
        actual_cost_tuple = (
            self.slippage_float,
            self.commission_per_share_float,
            self.commission_minimum_float,
        )
        if not all(
            np.isclose(actual_float, expected_float)
            for actual_float, expected_float in zip(
                actual_cost_tuple,
                expected_cost_tuple,
                strict=True,
            )
        ):
            raise ValueError("Trading costs do not match the accounting profile.")


DEFAULT_CONFIG = PassiveBilConfig()


def commission_float(
    share_count_int: int,
    commission_per_share_float: float,
    commission_minimum_float: float,
) -> float:
    """Return the engine's IBKR-style commission for a positive share count."""

    if share_count_int <= 0 or commission_per_share_float == 0.0:
        return 0.0
    return float(
        max(
            commission_minimum_float,
            commission_per_share_float * float(share_count_int),
        )
    )


def affordable_share_count_int(
    cash_float: float,
    sizing_close_float: float,
    slippage_float: float,
    commission_per_share_float: float,
    commission_minimum_float: float,
) -> int:
    """Largest whole-share target affordable under the known Close_T budget.

    The rule reserves the configured slippage and commission at ``Close_T``.
    A positive overnight gap can still make cash negative at ``Open_(T+1)``;
    that is an execution diagnostic, not silently prevented with future data.
    """

    if not np.isfinite(cash_float) or cash_float < 0.0:
        raise ValueError("cash_float must be finite and non-negative.")
    if not np.isfinite(sizing_close_float) or sizing_close_float <= 0.0:
        raise ValueError("sizing_close_float must be finite and positive.")

    conservative_unit_cost_float = float(
        sizing_close_float * (1.0 + slippage_float)
        + commission_per_share_float
    )
    candidate_share_count_int = int(cash_float // conservative_unit_cost_float)
    while candidate_share_count_int > 0:
        estimated_cost_float = float(
            candidate_share_count_int
            * sizing_close_float
            * (1.0 + slippage_float)
            + commission_float(
                candidate_share_count_int,
                commission_per_share_float,
                commission_minimum_float,
            )
        )
        if estimated_cost_float <= cash_float + 1e-12:
            return candidate_share_count_int
        candidate_share_count_int -= 1
    return 0


def canonical_pricing_sha256_str(pricing_data_df: pd.DataFrame) -> str:
    """Hash the exact BIL and benchmark rows consumed by this control."""

    canonical_data_df = pricing_data_df.sort_index().copy()
    canonical_data_df.index = canonical_data_df.index.strftime("%Y-%m-%d")
    canonical_csv_str = canonical_data_df.to_csv(
        index=True,
        na_rep="NA",
        float_format="%.12g",
        lineterminator="\n",
    )
    return hashlib.sha256(canonical_csv_str.encode("utf-8")).hexdigest()


def prepare_pricing_and_calendar(
    pricing_data_df: pd.DataFrame,
    backtest_start_date_str: str | None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Keep real BIL sessions and preserve a prior-close entry boundary."""

    required_column_tuple = (
        (ASSET_STR, "Open"),
        (ASSET_STR, "High"),
        (ASSET_STR, "Low"),
        (ASSET_STR, "Close"),
        (ASSET_STR, "Dividend"),
        (BENCHMARK_TUPLE[0], "Close"),
    )
    missing_column_list = [
        column_tuple
        for column_tuple in required_column_tuple
        if column_tuple not in pricing_data_df.columns
    ]
    if missing_column_list:
        raise ValueError(
            "Passive BIL pricing data is missing required columns: "
            f"{missing_column_list}."
        )
    if pricing_data_df.index.has_duplicates:
        raise ValueError("pricing_data_df index must be unique.")

    adjustment_by_symbol_dict = pricing_data_df.attrs.get(
        "norgate_adjustment_by_symbol_dict"
    )
    expected_adjustment_by_symbol_dict = {
        ASSET_STR: CAPITALSPECIAL_ADJUSTMENT_STR,
        BENCHMARK_TUPLE[0]: TOTALRETURN_ADJUSTMENT_STR,
    }
    if adjustment_by_symbol_dict != expected_adjustment_by_symbol_dict:
        raise ValueError(
            "Passive BIL requires explicit adjustment provenance "
            f"{expected_adjustment_by_symbol_dict}; found "
            f"{adjustment_by_symbol_dict!r}."
        )
    benchmark_data_symbol_dict = pricing_data_df.attrs.get(
        "benchmark_data_symbol_dict"
    )
    expected_benchmark_data_symbol_dict = {BENCHMARK_TUPLE[0]: "$SPXTR"}
    if benchmark_data_symbol_dict != expected_benchmark_data_symbol_dict:
        raise ValueError(
            "Passive BIL requires the genuine total-return benchmark mapping "
            f"{expected_benchmark_data_symbol_dict}; found "
            f"{benchmark_data_symbol_dict!r}."
        )

    sorted_pricing_data_df = pricing_data_df.sort_index().copy()
    original_attrs_dict = dict(pricing_data_df.attrs)
    required_numeric_column_list = [
        (ASSET_STR, "Open"),
        (ASSET_STR, "High"),
        (ASSET_STR, "Low"),
        (ASSET_STR, "Close"),
        (ASSET_STR, "Dividend"),
        (BENCHMARK_TUPLE[0], "Close"),
    ]
    numeric_required_df = sorted_pricing_data_df.loc[
        :, required_numeric_column_list
    ].apply(pd.to_numeric, errors="coerce")
    finite_session_mask_ser = pd.Series(
        np.isfinite(numeric_required_df.to_numpy(dtype=float)).all(axis=1),
        index=sorted_pricing_data_df.index,
    )
    if not finite_session_mask_ser.any():
        raise RuntimeError("Passive BIL has no complete real pricing session.")
    first_valid_position_int = int(np.flatnonzero(finite_session_mask_ser.to_numpy())[0])
    # *** CRITICAL*** Only leading pre-inception rows may be removed. Dropping an
    # internal row would compress time, lose a possible dividend entitlement,
    # and silently turn a multi-session return into a one-row transition.
    post_inception_valid_ser = finite_session_mask_ser.iloc[first_valid_position_int:]
    if not post_inception_valid_ser.all():
        invalid_date_list = post_inception_valid_ser.index[
            ~post_inception_valid_ser
        ].tolist()
        raise RuntimeError(
            "Passive BIL has an incomplete internal session after inception: "
            f"{invalid_date_list[:5]}."
        )
    valid_pricing_data_df = sorted_pricing_data_df.iloc[
        first_valid_position_int:
    ].copy()
    valid_pricing_data_df.attrs.update(original_attrs_dict)
    if len(valid_pricing_data_df) < 2:
        raise RuntimeError(
            "Passive BIL requires at least two real sessions: Close_T and Open_(T+1)."
        )

    requested_start_ts = (
        pd.Timestamp(valid_pricing_data_df.index[0])
        if backtest_start_date_str is None
        else pd.Timestamp(backtest_start_date_str)
    )
    first_execution_position_int = int(
        valid_pricing_data_df.index.searchsorted(requested_start_ts, side="left")
    )
    # *** CRITICAL*** If the request precedes BIL inception, the first real BIL
    # close is only a decision anchor. Entry occurs at the following real open;
    # no pre-inception return or synthetic cash row is created.
    first_execution_position_int = max(first_execution_position_int, 1)
    execution_calendar_idx = pd.DatetimeIndex(
        valid_pricing_data_df.index[first_execution_position_int:]
    )
    if len(execution_calendar_idx) == 0:
        raise RuntimeError("Passive BIL execution calendar is empty.")
    return valid_pricing_data_df, execution_calendar_idx


def load_passive_bil_pricing_data(
    config_obj: PassiveBilConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load CAPITALSPECIAL BIL and a TOTALRETURN $SPX reporting benchmark."""

    return load_raw_prices(
        symbols=[config_obj.asset_str],
        benchmarks=list(config_obj.benchmark_tuple),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )


@lru_cache(maxsize=1)
def load_tactical_fi_cash_contract_tuple() -> tuple[pd.Series, tuple[dict[str, str], ...]]:
    """Load the exact frozen DGS3MO cash series and provenance used by TFI."""

    from strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd import (
        DEFAULT_CONFIG as TACTICAL_FI_DEFAULT_CONFIG,
        get_tactical_yield_data,
    )

    (
        _execution_price_df,
        _yield_df,
        _signal_df,
        _rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = get_tactical_yield_data(TACTICAL_FI_DEFAULT_CONFIG)
    dgs3mo_snapshot_list = [
        snapshot_obj
        for snapshot_obj in fred_snapshot_tuple
        if snapshot_obj.series_id_str == "DGS3MO"
    ]
    if len(dgs3mo_snapshot_list) != 1:
        raise RuntimeError("Expected exactly one frozen DGS3MO snapshot.")
    snapshot_obj = dgs3mo_snapshot_list[0]
    provenance_tuple = (
        {
            "series_id_str": str(snapshot_obj.series_id_str),
            "source_path_str": str(snapshot_obj.source_path_str),
            "sha256_str": str(snapshot_obj.sha256_str),
            "latest_observation_date_str": (
                snapshot_obj.latest_observation_date_ts.date().isoformat()
            ),
            "vintage_policy_str": str(snapshot_obj.vintage_policy_str),
        },
    )
    return cash_return_ser.astype(float).copy(), provenance_tuple


class PassiveBilControlStrategy(Strategy):
    """Buy BIL once at the next open, then hold with dividends in cash."""

    def __init__(
        self,
        name: str = STRATEGY_NAME_STR,
        benchmarks: Sequence[str] | None = None,
        config_obj: PassiveBilConfig = DEFAULT_CONFIG,
        cash_return_ser: pd.Series | None = None,
        cash_provenance_tuple: tuple[dict[str, str], ...] = (),
    ) -> None:
        benchmark_list = list(
            config_obj.benchmark_tuple if benchmarks is None else benchmarks
        )
        super().__init__(
            name=name,
            benchmarks=benchmark_list,
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            performance_benchmark_symbol_str=benchmark_list[0],
            performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        )
        self.config_obj = config_obj
        self.entry_submitted_bool = False
        self.entry_decision_date_ts: pd.Timestamp | None = None
        self.entry_target_share_int = 0
        self.source_pricing_sha256_str: str | None = None
        self.cash_return_ser = (
            pd.Series(dtype=float)
            if cash_return_ser is None
            else cash_return_ser.astype(float).copy()
        )
        self.cash_interest_processed_date_set: set[pd.Timestamp] = set()
        self.cash_interest_ledger_row_dict_list: list[dict[str, object]] = []
        self.cash_interest_total_float = 0.0
        self.configure_dividend_cash_ledger(
            enabled_bool=True,
            withholding_rate_float=config_obj.dividend_withholding_rate_float,
        )
        self._data_adjustment_policy_dict.update(
            {
                "execution_and_marks_adjustment_str": (
                    CAPITALSPECIAL_ADJUSTMENT_STR
                ),
                "performance_benchmark_adjustment_str": (
                    TOTALRETURN_ADJUSTMENT_STR
                ),
            }
        )
        self._accounting_policy_dict.update(
            {
                "control_role_str": "research_only_passive_matched_control",
                "accounting_profile_str": config_obj.accounting_profile_str,
                "positive_cash_rate_policy_str": (
                    "causal_DGS3MO_ACT_365"
                    if config_obj.accounting_profile_str
                    == TACTICAL_FI_MATCHED_PROFILE_STR
                    else "zero_percent_intentional"
                ),
                "negative_cash_financing_policy_str": "not_modeled",
                "dividend_reinvestment_policy_str": "none_cash_accumulates",
                "entry_policy_str": "one_time_close_sized_next_open_fill",
                "rebalance_policy_str": "none",
            }
        )
        if cash_provenance_tuple:
            self._data_adjustment_policy_dict["cash_rate_provenance_list"] = [
                dict(provenance_dict) for provenance_dict in cash_provenance_tuple
            ]

    def _accrue_positive_cash_interest_float(self) -> float:
        if self.config_obj.accounting_profile_str != TACTICAL_FI_MATCHED_PROFILE_STR:
            return 0.0
        current_bar_ts = pd.Timestamp(self.current_bar)
        if current_bar_ts in self.cash_interest_processed_date_set:
            return 0.0
        if current_bar_ts not in self.cash_return_ser.index:
            raise RuntimeError(
                f"Passive BIL is missing causal DGS3MO cash return for "
                f"{current_bar_ts.date()}."
            )
        cash_return_float = float(self.cash_return_ser.loc[current_bar_ts])
        if not np.isfinite(cash_return_float):
            raise RuntimeError(
                f"Passive BIL has invalid DGS3MO cash return for "
                f"{current_bar_ts.date()}."
            )
        positive_cash_base_float = max(float(self.cash), 0.0)
        cash_interest_float = positive_cash_base_float * cash_return_float
        self.cash += cash_interest_float
        self.cash_interest_total_float += cash_interest_float
        self.cash_interest_processed_date_set.add(current_bar_ts)
        self.cash_interest_ledger_row_dict_list.append(
            {
                "date": current_bar_ts,
                "positive_cash_base_float": positive_cash_base_float,
                "cash_return_float": cash_return_float,
                "cash_interest_float": cash_interest_float,
            }
        )
        self._accounting_policy_dict["cash_interest_total_float"] = float(
            self.cash_interest_total_float
        )
        return cash_interest_float

    def get_cash_interest_ledger(self) -> pd.DataFrame:
        """Return the causal cash-interest rows with a stable empty schema."""

        return pd.DataFrame(
            self.cash_interest_ledger_row_dict_list,
            columns=CASH_INTEREST_LEDGER_COLUMN_TUPLE,
        )

    def iterate(
        self,
        _data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        _open_price_ser: pd.Series,
    ) -> None:
        self._accrue_positive_cash_interest_float()
        if self.entry_submitted_bool or close_row_ser is None:
            return

        sizing_close_float = float(close_row_ser[(self.config_obj.asset_str, "Close")])
        # *** CRITICAL*** The target share count uses only cash and BIL Close_T.
        # Open_(T+1) is intentionally ignored here and used only by the engine
        # when the already-fixed whole-share order is filled.
        target_share_int = affordable_share_count_int(
            cash_float=float(self.cash),
            sizing_close_float=sizing_close_float,
            slippage_float=self.config_obj.slippage_float,
            commission_per_share_float=(
                self.config_obj.commission_per_share_float
            ),
            commission_minimum_float=self.config_obj.commission_minimum_float,
        )
        self.entry_decision_date_ts = pd.Timestamp(self.previous_bar)
        self.entry_target_share_int = int(target_share_int)
        self.entry_submitted_bool = True
        if target_share_int > 0:
            self.order_target(
                self.config_obj.asset_str,
                target_share_int,
                trade_id=1,
            )

    def finalize(self, _current_data_df: pd.DataFrame) -> None:
        negative_cash_ser = self.results["cash"].astype(float)
        negative_cash_ser = negative_cash_ser.loc[negative_cash_ser < 0.0]
        self._accounting_policy_dict.update(
            {
                "entry_decision_date_str": (
                    None
                    if self.entry_decision_date_ts is None
                    else self.entry_decision_date_ts.date().isoformat()
                ),
                "entry_target_share_int": int(self.entry_target_share_int),
                "filled_transaction_count_int": int(len(self.get_transactions())),
                "negative_cash_day_count_int": int(len(negative_cash_ser)),
                "minimum_cash_float": float(self.results["cash"].astype(float).min()),
                "source_pricing_sha256_str": self.source_pricing_sha256_str,
            }
        )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = DEFAULT_CONFIG.history_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
    accounting_profile_str: str = CORE5_MATCHED_PROFILE_STR,
    cash_return_ser: pd.Series | None = None,
) -> PassiveBilControlStrategy:
    """Run the research-only engine-native passive BIL control."""

    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
        accounting_profile_str=str(accounting_profile_str),
        dividend_withholding_rate_float=(
            0.0
            if accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT
        ),
        slippage_float=(
            TACTICAL_FI_SLIPPAGE_FLOAT
            if accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else DEFAULT_SLIPPAGE_FLOAT
        ),
        commission_per_share_float=(
            TACTICAL_FI_COMMISSION_PER_SHARE_FLOAT
            if accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else DEFAULT_COMMISSION_PER_SHARE_FLOAT
        ),
        commission_minimum_float=(
            TACTICAL_FI_COMMISSION_MINIMUM_FLOAT
            if accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR
            else DEFAULT_COMMISSION_MINIMUM_FLOAT
        ),
    )
    if pricing_data_df is None:
        pricing_data_df = load_passive_bil_pricing_data(config_obj=config_obj)
    else:
        pricing_data_df = pricing_data_df.copy()
        pricing_data_df.attrs.update(dict(pricing_data_df.attrs))
        if end_date_str is not None:
            # *** CRITICAL*** A test or research caller may inject a longer panel;
            # the frozen endpoint must still be enforced before any return is seen.
            injected_attrs_dict = dict(pricing_data_df.attrs)
            pricing_data_df = pricing_data_df.loc[
                pricing_data_df.index <= pd.Timestamp(end_date_str)
            ].copy()
            pricing_data_df.attrs.update(injected_attrs_dict)
    cash_provenance_tuple: tuple[dict[str, str], ...] = ()
    if accounting_profile_str == TACTICAL_FI_MATCHED_PROFILE_STR:
        if cash_return_ser is None:
            cash_return_ser, cash_provenance_tuple = (
                load_tactical_fi_cash_contract_tuple()
            )
        cash_return_ser = cash_return_ser.astype(float).copy()
        if end_date_str is not None:
            cash_return_ser = cash_return_ser.loc[
                cash_return_ser.index <= pd.Timestamp(end_date_str)
            ]
    elif cash_return_ser is not None:
        raise ValueError(
            "cash_return_ser is allowed only for the Tactical-FI-matched profile."
        )
    valid_pricing_data_df, execution_calendar_idx = prepare_pricing_and_calendar(
        pricing_data_df=pricing_data_df,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj = PassiveBilControlStrategy(
        config_obj=config_obj,
        cash_return_ser=cash_return_ser,
        cash_provenance_tuple=cash_provenance_tuple,
    )
    strategy_obj.source_pricing_sha256_str = canonical_pricing_sha256_str(
        valid_pricing_data_df
    )
    run_daily(
        strategy_obj,
        valid_pricing_data_df,
        calendar=execution_calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.get_transactions())
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


if __name__ == "__main__":
    run_variant()
