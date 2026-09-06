"""Frozen VIXM-in-backwardation hedge strategy.

After each official ``Close_T``:

    target_VIXM,T = 1 if VIX_T > VIX3M_T else 0
    target_SHY,T = 1 - target_VIXM,T

The strict threshold is the entire signal. Orders are sized after ``Close_T``
and fill at ``Open_(T+1)``. There is no VIX-level sizing, eVRP filter, moving
average, holding-period rule, stop, or VIXY/VXX substitution.

The frozen Pakal report treats SHY as a return-bearing cash proxy, charges
turnover only on the VIXM target, and reports Open-to-Open returns. This Alpha
path holds SHY as a real traded asset, charges both executed legs, and marks
NAV at each close. Its signal and target state can match the frozen artifact,
but its performance metrics are intentionally not labeled exact parity.

This module is research/BENCH only. It contains no LIVE, broker, scheduler,
release, or allocation wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


STRATEGY_NAME_STR = "strategy_vixm_backwardation"
STRATEGY_DISPLAY_NAME_STR = "VIXM Backwardation"
VIXM_ASSET_STR = "VIXM"
RESERVE_ASSET_STR = "SHY"
TRADEABLE_ASSET_TUPLE = (VIXM_ASSET_STR, RESERVE_ASSET_STR)
BENCHMARK_TUPLE = ("$SPX",)
VIX_INDEX_STR = "$VIX"
VIX3M_INDEX_STR = "$VIX3M"
VIX_SIGNAL_NAMESPACE_STR = "VIX_BACKWARDATION"
STATE_FIELD_STR = "backwardation_state_float"
TERM_RATIO_FIELD_STR = "vix_to_vix3m_ratio_float"
ONE_WAY_COST_FLOAT = 0.001


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class VixmBackwardationConfig:
    strategy_name_str: str = STRATEGY_NAME_STR
    history_start_date_str: str = "2011-02-01"
    backtest_start_date_str: str = "2011-03-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.strategy_name_str:
            raise ValueError("strategy_name_str must not be empty.")
        if self.capital_base_float <= 0.0 or not np.isfinite(self.capital_base_float):
            raise ValueError("capital_base_float must be positive and finite.")


DEFAULT_CONFIG = VixmBackwardationConfig()


def compute_backwardation_state_ser(
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
) -> pd.Series:
    """Return 1 only when both closes exist and VIX is strictly above VIX3M."""

    aligned_vix_close_ser, aligned_vix3m_close_ser = vix_close_ser.align(
        vix3m_close_ser,
        join="outer",
    )
    valid_input_bool_ser = (
        np.isfinite(aligned_vix_close_ser)
        & np.isfinite(aligned_vix3m_close_ser)
        & aligned_vix_close_ser.gt(0.0)
        & aligned_vix3m_close_ser.gt(0.0)
    )
    # *** CRITICAL*** The state uses only the two official closes stamped T.
    # Equality is contango/cash; no later open or future index value is used.
    backwardation_state_ser = (
        aligned_vix_close_ser > aligned_vix3m_close_ser
    ).astype(float).where(valid_input_bool_ser)
    backwardation_state_ser.name = STATE_FIELD_STR
    return backwardation_state_ser


def build_vixm_target_weight_df(
    backwardation_state_ser: pd.Series,
) -> pd.DataFrame:
    invalid_state_ser = backwardation_state_ser.dropna().loc[
        ~backwardation_state_ser.dropna().isin([0.0, 1.0])
    ]
    if len(invalid_state_ser) > 0:
        raise ValueError("backwardation_state_ser must contain only 0, 1, or NaN.")
    return pd.DataFrame(
        {
            VIXM_ASSET_STR: backwardation_state_ser,
            RESERVE_ASSET_STR: 1.0 - backwardation_state_ser,
        },
        index=backwardation_state_ser.index,
        dtype=float,
    )


def get_vixm_backwardation_data(
    config_obj: VixmBackwardationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load CAPITALSPECIAL execution bars and raw VIX/VIX3M closes."""

    execution_price_df = load_raw_prices(
        symbols=list(TRADEABLE_ASSET_TUPLE),
        benchmarks=list(BENCHMARK_TUPLE),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    index_price_df = load_raw_prices(
        symbols=[],
        benchmarks=[VIX_INDEX_STR, VIX3M_INDEX_STR],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    index_close_df = index_price_df.loc[
        :,
        [(VIX_INDEX_STR, "Close"), (VIX3M_INDEX_STR, "Close")],
    ].copy()
    index_close_df.columns = pd.MultiIndex.from_tuples(
        [
            (VIX_SIGNAL_NAMESPACE_STR, "vix_close_float"),
            (VIX_SIGNAL_NAMESPACE_STR, "vix3m_close_float"),
        ]
    )
    pricing_data_df = pd.concat([execution_price_df, index_close_df], axis=1).sort_index()
    vixm_calendar_bool_ser = pricing_data_df[(VIXM_ASSET_STR, "Close")].notna()
    pricing_data_df = pricing_data_df.loc[vixm_calendar_bool_ser].copy()
    pricing_data_df.attrs.update(execution_price_df.attrs)
    adjustment_by_symbol_dict = dict(
        pricing_data_df.attrs.get("norgate_adjustment_by_symbol_dict", {})
    )
    adjustment_by_symbol_dict[VIX_SIGNAL_NAMESPACE_STR] = "RAW_INDEX_CLOSE"
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = (
        adjustment_by_symbol_dict
    )
    pricing_data_df.attrs["signal_adjustment_by_symbol_dict"] = {
        VIX_SIGNAL_NAMESPACE_STR: "RAW_INDEX_CLOSE"
    }
    return pricing_data_df


class VixmBackwardationStrategy(Strategy):
    """Daily all-VIXM or all-SHY causal hedge sleeve."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        config_obj: VixmBackwardationConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__(
            name=config_obj.strategy_name_str,
            benchmarks=list(BENCHMARK_TUPLE),
            capital_base=config_obj.capital_base_float,
            slippage=ONE_WAY_COST_FLOAT,
            commission_per_share=0.0,
            commission_minimum=0.0,
            performance_benchmark_symbol_str=BENCHMARK_TUPLE[0],
            performance_benchmark_adjustment_str="TOTALRETURN",
        )
        self.config_obj = config_obj
        self.asset_list = list(TRADEABLE_ASSET_TUPLE)
        self.initialized_bool = False
        self.last_state_float = np.nan
        self.trade_id_int = 0
        self.current_trade_id_map = {
            asset_str: default_trade_id_int() for asset_str in self.asset_list
        }
        self.daily_target_weight_row_dict_list: list[dict[str, object]] = []
        self.rebalance_target_weight_row_dict_list: list[dict[str, object]] = []
        self.signal_diagnostic_df = pd.DataFrame()
        self.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=0.0)
        self._data_adjustment_policy_dict.update(
            {
                "signal_adjustment_str": "RAW_INDEX_CLOSE",
                "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                "performance_benchmark_adjustment_str": "TOTALRETURN",
            }
        )
        self._accounting_policy_dict.update(
            {
                "reserve_asset_str": RESERVE_ASSET_STR,
                "state_transition_execution_str": "Close_T_to_Open_(T+1)",
                "missing_signal_policy_str": "fail_loud",
                "reserve_execution_str": "physical_SHY_traded_with_slippage",
                "reference_reserve_str": "synthetic_SHY_cash_proxy",
                "performance_parity_status_str": "not_exact_open_to_open_parity",
            }
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        vix_close_ser = pricing_data_df[
            (VIX_SIGNAL_NAMESPACE_STR, "vix_close_float")
        ].astype(float)
        vix3m_close_ser = pricing_data_df[
            (VIX_SIGNAL_NAMESPACE_STR, "vix3m_close_float")
        ].astype(float)
        backwardation_state_ser = compute_backwardation_state_ser(
            vix_close_ser,
            vix3m_close_ser,
        )
        term_ratio_ser = vix_close_ser / vix3m_close_ser.replace(0.0, np.nan)
        feature_df = pd.DataFrame(
            {
                (VIX_SIGNAL_NAMESPACE_STR, STATE_FIELD_STR): backwardation_state_ser,
                (VIX_SIGNAL_NAMESPACE_STR, TERM_RATIO_FIELD_STR): term_ratio_ser,
            },
            index=pricing_data_df.index,
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        if len(feature_df) >= len(self.signal_diagnostic_df):
            self.signal_diagnostic_df = pd.DataFrame(
                {
                    "vix_close_float": vix_close_ser,
                    "vix3m_close_float": vix3m_close_ser,
                    TERM_RATIO_FIELD_STR: term_ratio_ser,
                    STATE_FIELD_STR: backwardation_state_ser,
                }
            )
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def _new_trade_id_int(self, asset_str: str) -> int:
        self.trade_id_int += 1
        self.current_trade_id_map[asset_str] = self.trade_id_int
        return self.trade_id_int

    def _filled_position_trade_id_int(
        self,
        asset_str: str,
        current_share_int: int,
    ) -> int:
        if current_share_int == 0:
            return default_trade_id_int()
        asset_transaction_df = self.get_transactions()
        asset_transaction_df = asset_transaction_df.loc[
            asset_transaction_df["asset"] == asset_str
        ]
        if len(asset_transaction_df) == 0:
            raise RuntimeError(f"Open {asset_str} position has no filled transaction.")
        return int(asset_transaction_df.iloc[-1]["trade_id"])

    def _submit_target_orders(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> None:
        current_position_ser = self.get_positions().reindex(
            self.asset_list,
            fill_value=0.0,
        ).astype(int)
        budget_value_float = float(self.previous_total_value)
        if not np.isfinite(budget_value_float) or budget_value_float <= 0.0:
            raise RuntimeError("Previous VIXM sleeve value must be positive.")

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            close_price_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid execution close for {asset_str} on {self.previous_bar}."
                )

            # *** CRITICAL*** Target shares are fixed using Close_T and prior
            # NAV. The next open supplies only the fill price.
            target_share_int = int(
                budget_value_float * target_weight_float / close_price_float
            )
            if target_share_int == current_share_int:
                continue
            if target_share_int == 0:
                trade_id_int = self._filled_position_trade_id_int(
                    asset_str,
                    current_share_int,
                )
                self.order_target(asset_str, 0, trade_id=trade_id_int)
                self.current_trade_id_map[asset_str] = default_trade_id_int()
                continue
            trade_id_int = (
                self._new_trade_id_int(asset_str)
                if current_share_int == 0
                else self._filled_position_trade_id_int(asset_str, current_share_int)
            )
            self.order_target(
                asset_str,
                target_share_int,
                trade_id=trade_id_int,
            )

    def _position_matches_state_bool(self, state_float: float) -> bool:
        """Return whether the filled holdings implement the current one-hot state."""

        current_position_ser = self.get_positions().reindex(
            self.asset_list,
            fill_value=0.0,
        )
        target_asset_str = VIXM_ASSET_STR if state_float == 1.0 else RESERVE_ASSET_STR
        other_asset_str = RESERVE_ASSET_STR if state_float == 1.0 else VIXM_ASSET_STR
        return bool(
            current_position_ser.loc[target_asset_str] > 0.0
            and current_position_ser.loc[other_asset_str] == 0.0
        )

    def _record_target_weight(
        self,
        target_weight_ser: pd.Series,
        rebalance_bool: bool,
    ) -> None:
        target_record_dict = {
            "decision_date_ts": pd.Timestamp(self.previous_bar),
            VIXM_ASSET_STR: float(target_weight_ser.loc[VIXM_ASSET_STR]),
            RESERVE_ASSET_STR: float(target_weight_ser.loc[RESERVE_ASSET_STR]),
        }
        self.daily_target_weight_row_dict_list.append(target_record_dict)
        if rebalance_bool:
            self.rebalance_target_weight_row_dict_list.append(target_record_dict.copy())

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        _open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return
        state_float = float(
            close_row_ser.get(
                (VIX_SIGNAL_NAMESPACE_STR, STATE_FIELD_STR),
                np.nan,
            )
        )
        if state_float not in (0.0, 1.0):
            raise RuntimeError(
                f"Missing VIX/VIX3M state at Close_{self.previous_bar}."
            )
        state_changed_bool = bool(
            self.initialized_bool and state_float != self.last_state_float
        )
        # A transition order can be canceled when the next open is missing.
        # Compare the filled position with the desired one-hot state every day,
        # so a canceled leg is retried instead of losing the episode silently.
        position_mismatch_bool = not self._position_matches_state_bool(state_float)
        rebalance_bool = bool(
            not self.initialized_bool or state_changed_bool or position_mismatch_bool
        )
        target_weight_ser = pd.Series(
            {
                VIXM_ASSET_STR: state_float,
                RESERVE_ASSET_STR: 1.0 - state_float,
            },
            dtype=float,
        )
        if rebalance_bool:
            self._submit_target_orders(target_weight_ser, close_row_ser)
        self._record_target_weight(target_weight_ser, rebalance_bool)
        self.last_state_float = state_float
        self.initialized_bool = True

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        if self.daily_target_weight_row_dict_list:
            self.daily_target_weights = pd.DataFrame(
                self.daily_target_weight_row_dict_list
            ).set_index("decision_date_ts").sort_index()
        if self.rebalance_target_weight_row_dict_list:
            self.rebalance_target_weight_df = pd.DataFrame(
                self.rebalance_target_weight_row_dict_list
            ).set_index("decision_date_ts").sort_index()


def build_execution_calendar_idx(
    pricing_data_df: pd.DataFrame,
    backtest_start_date_str: str,
) -> pd.DatetimeIndex:
    required_signal_df = pricing_data_df.loc[
        :,
        [
            (VIX_SIGNAL_NAMESPACE_STR, "vix_close_float"),
            (VIX_SIGNAL_NAMESPACE_STR, "vix3m_close_float"),
        ],
    ].astype(float)
    valid_signal_bool_ser = np.isfinite(required_signal_df).all(axis=1) & (
        required_signal_df > 0.0
    ).all(axis=1)
    valid_signal_position_vec = np.flatnonzero(valid_signal_bool_ser.to_numpy())
    valid_execution_position_vec = valid_signal_position_vec + 1
    valid_execution_position_vec = valid_execution_position_vec[
        valid_execution_position_vec < len(pricing_data_df.index)
    ]
    if len(valid_execution_position_vec) == 0:
        raise RuntimeError("No actionable VIXM execution session is available.")
    first_valid_execution_ts = pd.Timestamp(
        pricing_data_df.index[valid_execution_position_vec[0]]
    )
    calendar_start_ts = max(
        first_valid_execution_ts,
        pd.Timestamp(backtest_start_date_str),
    )
    calendar_idx = pd.DatetimeIndex(
        pricing_data_df.index[pricing_data_df.index >= calendar_start_ts]
    )
    if len(calendar_idx) < 2:
        raise RuntimeError("VIXM execution calendar requires at least two sessions.")
    return calendar_idx


def run_variant(
    show_display_bool: bool = False,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
) -> VixmBackwardationStrategy:
    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else backtest_start_date_str
        ),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_vixm_backwardation_data(config_obj)
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df,
        config_obj.backtest_start_date_str,
    )
    strategy_obj = VixmBackwardationStrategy(config_obj)
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )
    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


if __name__ == "__main__":
    run_variant(show_display_bool=True, save_results_bool=True)
