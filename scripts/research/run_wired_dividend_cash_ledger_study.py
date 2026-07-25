"""Research-only A/B study for explicit dividend cash in WIRED strategies.

The baseline is the unchanged engine ledger. The candidate credits or debits
cash on ex-date before open orders, using the position held at the prior
session close and Norgate's prior-session Dividend field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy


WIRED_STRATEGY_MODULE_TUPLE = (
    "strategies.dv2.strategy_mr_dv2",
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "strategies.momentum.strategy_mo_atr_normalized_ndx",
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled",
)
DIVIDEND_LEDGER_COLUMN_TUPLE = (
    "entitlement_date",
    "ex_date",
    "asset_str",
    "position_share_float",
    "dividend_per_share_float",
    "gross_dividend_cash_float",
    "withholding_cash_float",
    "net_dividend_cash_float",
)
SUMMARY_METRIC_MAP_DICT = {
    "cagr_pct_float": "Return (Ann.) [%]",
    "annualized_volatility_pct_float": "Volatility (Ann.) [%]",
    "sharpe_float": "Sharpe Ratio",
    "max_drawdown_pct_float": "Max. Drawdown [%]",
    "mar_float": "MAR Ratio",
    "turnover_ann_pct_float": "Turnover (Ann.) [%]",
    "total_commissions_float": "Total Commissions [$]",
    "slippage_cost_float": "Slippage Cost [$]",
}
SIGNAL_DIAGNOSTIC_ATTRIBUTE_TUPLE = (
    "daily_target_weights",
    "rebalance_schedule_df",
    "vxn_scale_signal_df",
    "daily_vrp_signal_df",
    "month_end_vrp_diagnostic_df",
)
_PROCESS_ORDERS_PATCH_LOCK = threading.Lock()


def _validated_withholding_rate_float(withholding_rate_float: float) -> float:
    validated_rate_float = float(withholding_rate_float)
    if not np.isfinite(validated_rate_float):
        raise ValueError("withholding_rate_float must be finite.")
    if validated_rate_float < 0.0 or validated_rate_float > 1.0:
        raise ValueError("withholding_rate_float must be between 0 and 1.")
    return validated_rate_float


def _initialize_dividend_ledger_state(
    strategy_obj: Strategy,
    *,
    withholding_rate_float: float,
) -> None:
    if hasattr(strategy_obj, "_research_dividend_processed_ex_date_set"):
        return

    strategy_obj._research_dividend_processed_ex_date_set = set()
    strategy_obj._research_dividend_ledger_row_dict_list = []
    strategy_obj.dividend_cash_gross_total_float = 0.0
    strategy_obj.dividend_withholding_total_float = 0.0
    strategy_obj.dividend_cash_net_total_float = 0.0
    strategy_obj.dividend_withholding_rate_float = float(withholding_rate_float)
    strategy_obj._accounting_policy_dict = {
        **getattr(strategy_obj, "_accounting_policy_dict", {}),
        "accounting_contract_version_str": "research_dividend_cash_ledger_v1",
        "dividend_policy_str": "explicit_ex_date_cash_no_reinvestment",
        "dividend_withholding_rate_float": float(withholding_rate_float),
        "positive_cash_rate_policy_str": "zero_percent_intentional",
    }


def get_dividend_ledger_df(strategy_obj: Strategy) -> pd.DataFrame:
    """Return the research dividend ledger with a stable empty schema."""
    ledger_row_dict_list = getattr(
        strategy_obj,
        "_research_dividend_ledger_row_dict_list",
        [],
    )
    return pd.DataFrame(
        ledger_row_dict_list,
        columns=DIVIDEND_LEDGER_COLUMN_TUPLE,
    )


def credit_dividend_cash_before_open(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    *,
    withholding_rate_float: float = 0.0,
) -> float:
    """Credit prior-close entitlement cash before the current session open.

    For asset i and ex-date T+1:

        gross_dividend_cash_{i,T+1}
            = shares_held_at_close_{i,T} * Dividend_{i,T}

        withholding_{i,T+1}
            = max(gross_dividend_cash_{i,T+1}, 0) * withholding_rate

        net_dividend_cash_{i,T+1}
            = gross_dividend_cash_{i,T+1} - withholding_{i,T+1}

    A negative position therefore pays the full manufactured dividend. No
    shares are purchased by this function; cash remains cash until ordinary
    future strategy orders use account NAV.
    """
    validated_rate_float = _validated_withholding_rate_float(
        withholding_rate_float
    )
    _initialize_dividend_ledger_state(
        strategy_obj,
        withholding_rate_float=validated_rate_float,
    )

    if strategy_obj.current_bar is None or strategy_obj.previous_bar is None:
        return 0.0

    ex_date_ts = pd.Timestamp(strategy_obj.current_bar)
    entitlement_date_ts = pd.Timestamp(strategy_obj.previous_bar)
    processed_ex_date_set = strategy_obj._research_dividend_processed_ex_date_set
    if ex_date_ts in processed_ex_date_set:
        return 0.0
    if ex_date_ts not in pricing_data_df.index:
        return 0.0
    if entitlement_date_ts not in pricing_data_df.index:
        raise RuntimeError(
            "Dividend entitlement date is missing from pricing_data_df: "
            f"{entitlement_date_ts.date()}."
        )

    # *** CRITICAL*** Norgate stamps Dividend on entitlement session T. The
    # cash event belongs to the next market session T+1, before that session's
    # open orders. Positions are sampled before those orders, so an ex-date
    # buyer gets nothing and an ex-date seller keeps the earned distribution.
    preopen_position_ser = strategy_obj.get_positions().astype(float)
    active_position_ser = preopen_position_ser.loc[
        ~np.isclose(preopen_position_ser, 0.0)
    ]

    pending_ledger_row_dict_list: list[dict[str, object]] = []
    gross_dividend_cash_sum_float = 0.0
    withholding_cash_sum_float = 0.0
    net_dividend_cash_sum_float = 0.0
    for asset_str, position_share_float in active_position_ser.items():
        dividend_column_tuple = (str(asset_str), "Dividend")
        if dividend_column_tuple not in pricing_data_df.columns:
            raise RuntimeError(
                f"Missing Norgate Dividend field for active asset {asset_str}."
            )

        dividend_value_obj = pricing_data_df.loc[
            entitlement_date_ts,
            dividend_column_tuple,
        ]
        dividend_per_share_float = float(
            pd.to_numeric(pd.Series([dividend_value_obj]), errors="coerce").iloc[0]
        )
        if not np.isfinite(dividend_per_share_float):
            raise RuntimeError(
                "Invalid Dividend for active asset "
                f"{asset_str} on entitlement date {entitlement_date_ts.date()}."
            )
        if np.isclose(dividend_per_share_float, 0.0):
            continue

        gross_dividend_cash_float = (
            float(position_share_float) * dividend_per_share_float
        )
        withholding_cash_float = (
            max(gross_dividend_cash_float, 0.0) * validated_rate_float
        )
        net_dividend_cash_float = (
            gross_dividend_cash_float - withholding_cash_float
        )
        net_dividend_cash_sum_float += net_dividend_cash_float
        gross_dividend_cash_sum_float += gross_dividend_cash_float
        withholding_cash_sum_float += withholding_cash_float
        pending_ledger_row_dict_list.append(
            {
                "entitlement_date": entitlement_date_ts,
                "ex_date": ex_date_ts,
                "asset_str": str(asset_str),
                "position_share_float": float(position_share_float),
                "dividend_per_share_float": dividend_per_share_float,
                "gross_dividend_cash_float": gross_dividend_cash_float,
                "withholding_cash_float": withholding_cash_float,
                "net_dividend_cash_float": net_dividend_cash_float,
            }
        )

    # Apply the event atomically only after every active asset has a valid
    # Dividend field. A later validation error must not leave a partial ledger.
    strategy_obj.cash += net_dividend_cash_sum_float
    strategy_obj.dividend_cash_gross_total_float += (
        gross_dividend_cash_sum_float
    )
    strategy_obj.dividend_withholding_total_float += withholding_cash_sum_float
    strategy_obj.dividend_cash_net_total_float += net_dividend_cash_sum_float
    strategy_obj._research_dividend_ledger_row_dict_list.extend(
        pending_ledger_row_dict_list
    )
    processed_ex_date_set.add(ex_date_ts)
    return net_dividend_cash_sum_float


@contextmanager
def research_dividend_cash_ledger_context(
    *,
    withholding_rate_float: float = 0.0,
) -> Iterator[None]:
    """Temporarily wrap the shared order processor for one research run."""
    validated_rate_float = _validated_withholding_rate_float(
        withholding_rate_float
    )
    lock_acquired_bool = _PROCESS_ORDERS_PATCH_LOCK.acquire(blocking=False)
    if not lock_acquired_bool:
        raise RuntimeError(
            "Research dividend wrapper is already active. "
            "Run this study only in one exclusive serial CLI process."
        )

    original_process_orders_fn = Strategy.process_orders

    def dividend_aware_process_orders_fn(
        strategy_obj: Strategy,
        prices: pd.DataFrame,
    ):
        pricing_data_df = prices
        credit_dividend_cash_before_open(
            strategy_obj,
            pricing_data_df,
            withholding_rate_float=validated_rate_float,
        )
        return original_process_orders_fn(strategy_obj, pricing_data_df)

    Strategy.process_orders = dividend_aware_process_orders_fn
    try:
        yield
    finally:
        Strategy.process_orders = original_process_orders_fn
        _PROCESS_ORDERS_PATCH_LOCK.release()


def _assert_pair_inputs_equal(
    baseline_input_dict: dict[str, object],
    dividend_input_dict: dict[str, object],
) -> None:
    baseline_pricing_data_df = baseline_input_dict["pricing_data_df"]
    dividend_pricing_data_df = dividend_input_dict["pricing_data_df"]
    if not isinstance(baseline_pricing_data_df, pd.DataFrame):
        raise TypeError("Baseline pricing_data_df must be a DataFrame.")
    if not isinstance(dividend_pricing_data_df, pd.DataFrame):
        raise TypeError("Dividend pricing_data_df must be a DataFrame.")
    if not baseline_pricing_data_df.index.equals(dividend_pricing_data_df.index):
        raise RuntimeError("A/B pricing calendars differ.")
    if not baseline_pricing_data_df.columns.equals(
        dividend_pricing_data_df.columns
    ):
        raise RuntimeError("A/B pricing columns differ.")
    if not baseline_pricing_data_df.equals(dividend_pricing_data_df):
        raise RuntimeError(
            "A/B pricing values differ. The dividend ledger is not the only "
            "changed input."
        )

    baseline_strategy_obj = baseline_input_dict["strategy_obj"]
    dividend_strategy_obj = dividend_input_dict["strategy_obj"]
    if not baseline_strategy_obj.results.index.equals(
        dividend_strategy_obj.results.index
    ):
        raise RuntimeError("A/B result calendars differ.")

    for attribute_name_str in SIGNAL_DIAGNOSTIC_ATTRIBUTE_TUPLE:
        baseline_value_obj = getattr(
            baseline_strategy_obj,
            attribute_name_str,
            None,
        )
        dividend_value_obj = getattr(
            dividend_strategy_obj,
            attribute_name_str,
            None,
        )
        if baseline_value_obj is None and dividend_value_obj is None:
            continue
        if type(baseline_value_obj) is not type(dividend_value_obj):
            raise RuntimeError(
                f"A/B diagnostic {attribute_name_str} has different types."
            )
        if isinstance(baseline_value_obj, (pd.DataFrame, pd.Series)):
            if not baseline_value_obj.equals(dividend_value_obj):
                raise RuntimeError(
                    f"A/B diagnostic {attribute_name_str} differs."
                )


def _resolve_parity_artifact(
    parity_artifact_path_str: str | None,
    *,
    output_dir_str: str,
) -> tuple[Path, str]:
    if parity_artifact_path_str is None:
        parity_root_path = (
            Path(output_dir_str)
            / "research"
            / "accounting"
            / "dividend_total_return_parity"
            / "parity_study"
        )
        candidate_path_list = sorted(
            (
                candidate_path
                for candidate_path in parity_root_path.glob("*")
                if (candidate_path / "metadata.json").is_file()
            ),
            reverse=True,
        )
        if len(candidate_path_list) == 0:
            raise RuntimeError(
                "No dividend parity artifact was found. Run the parity study first."
            )
        parity_artifact_path = candidate_path_list[0]
    else:
        parity_artifact_path = Path(parity_artifact_path_str)

    metadata_path = parity_artifact_path / "metadata.json"
    symbol_summary_path = parity_artifact_path / "symbol_summary.csv"
    if not metadata_path.is_file() or not symbol_summary_path.is_file():
        raise RuntimeError(
            f"Parity artifact is incomplete: {parity_artifact_path}."
        )
    metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_dict.get("overall_pass_bool") is not True:
        raise RuntimeError(
            f"Parity artifact did not pass: {parity_artifact_path}."
        )

    parity_hash_obj = hashlib.sha256()
    parity_hash_obj.update(metadata_path.read_bytes())
    parity_hash_obj.update(symbol_summary_path.read_bytes())
    return parity_artifact_path.resolve(), parity_hash_obj.hexdigest()


def _builder_keyword_dict(
    builder_fn: Callable[..., dict[str, object]],
    *,
    backtest_start_date_str: str | None,
    capital_base_float: float,
    end_date_str: str | None,
    show_progress_bool: bool,
) -> dict[str, object]:
    requested_keyword_dict = {
        "show_display_bool": bool(show_progress_bool),
        "backtest_start_date_str": backtest_start_date_str,
        "capital_base_float": float(capital_base_float),
        "end_date_str": end_date_str,
    }
    signature_obj = inspect.signature(builder_fn)
    return {
        keyword_str: value_obj
        for keyword_str, value_obj in requested_keyword_dict.items()
        if keyword_str in signature_obj.parameters and value_obj is not None
    }


def _summary_metric_float(
    summary_df: pd.DataFrame,
    metric_name_str: str,
) -> float:
    if metric_name_str not in summary_df.index:
        return np.nan
    metric_value_obj = summary_df.loc[metric_name_str, "Strategy"]
    return float(pd.to_numeric(pd.Series([metric_value_obj]), errors="coerce").iloc[0])


def _execution_skeleton_df(strategy_obj: Strategy) -> pd.DataFrame:
    transaction_df = strategy_obj.get_transactions().copy()
    if len(transaction_df) == 0:
        return pd.DataFrame(
            columns=["bar", "asset", "direction_int"]
        )
    transaction_df["direction_int"] = np.sign(
        pd.to_numeric(transaction_df["amount"], errors="coerce")
    ).astype(int)
    return transaction_df.loc[
        :,
        ["bar", "asset", "direction_int"],
    ].reset_index(drop=True)


def _cash_diagnostic_dict(
    cash_ser: pd.Series,
    total_value_ser: pd.Series,
    *,
    prefix_str: str,
) -> dict[str, object]:
    cash_float_ser = cash_ser.astype(float)
    total_value_float_ser = total_value_ser.astype(float)
    cash_weight_ser = cash_float_ser / total_value_float_ser
    negative_cash_mask_ser = cash_float_ser.lt(-1e-8)
    # *** CRITICAL*** Diagnostic-only backward lag: shift(1) asks whether the
    # immediately prior simulated session was already negative. It never uses
    # a future cash observation and cannot enter signal or order logic.
    negative_cash_start_mask_ser = (
        negative_cash_mask_ser
        & ~negative_cash_mask_ser.shift(1, fill_value=False)
    )
    negative_cash_float_ser = cash_float_ser.loc[negative_cash_mask_ser]
    negative_cash_weight_ser = cash_weight_ser.loc[negative_cash_mask_ser]

    return {
        f"{prefix_str}_negative_cash_day_count_int": int(
            negative_cash_mask_ser.sum()
        ),
        f"{prefix_str}_negative_cash_day_fraction_float": float(
            negative_cash_mask_ser.mean()
        ),
        f"{prefix_str}_negative_cash_episode_count_int": int(
            negative_cash_start_mask_ser.sum()
        ),
        f"{prefix_str}_minimum_cash_float": float(cash_float_ser.min()),
        f"{prefix_str}_minimum_cash_weight_float": float(cash_weight_ser.min()),
        f"{prefix_str}_average_negative_cash_float": (
            0.0
            if len(negative_cash_float_ser) == 0
            else float(negative_cash_float_ser.mean())
        ),
        f"{prefix_str}_average_negative_cash_weight_float": (
            0.0
            if len(negative_cash_weight_ser) == 0
            else float(negative_cash_weight_ser.mean())
        ),
    }


def _strategy_result_row_dict(
    strategy_module_str: str,
    *,
    baseline_strategy_obj: Strategy,
    dividend_strategy_obj: Strategy,
) -> dict[str, object]:
    baseline_cash_ser = baseline_strategy_obj.results["cash"].astype(float)
    dividend_cash_ser = dividend_strategy_obj.results["cash"].astype(float)
    baseline_total_value_ser = baseline_strategy_obj.results[
        "total_value"
    ].astype(float)
    dividend_total_value_ser = dividend_strategy_obj.results[
        "total_value"
    ].astype(float)
    executed_transaction_skeleton_equal_bool = bool(
        _execution_skeleton_df(baseline_strategy_obj).equals(
            _execution_skeleton_df(dividend_strategy_obj)
        )
    )
    baseline_cash_diagnostic_dict = _cash_diagnostic_dict(
        baseline_cash_ser,
        baseline_total_value_ser,
        prefix_str="baseline",
    )
    dividend_cash_diagnostic_dict = _cash_diagnostic_dict(
        dividend_cash_ser,
        dividend_total_value_ser,
        prefix_str="dividend",
    )
    baseline_negative_cash_day_count_int = int(
        baseline_cash_diagnostic_dict[
            "baseline_negative_cash_day_count_int"
        ]
    )
    dividend_negative_cash_day_count_int = int(
        dividend_cash_diagnostic_dict[
            "dividend_negative_cash_day_count_int"
        ]
    )
    negative_cash_breach_bool = bool(
        baseline_negative_cash_day_count_int > 0
        or dividend_negative_cash_day_count_int > 0
    )
    result_row_dict: dict[str, object] = {
        "strategy_module_str": strategy_module_str,
        "start_date_str": str(dividend_strategy_obj.results.index[0]),
        "end_date_str": str(dividend_strategy_obj.results.index[-1]),
        "baseline_terminal_value_float": float(
            baseline_strategy_obj.results["total_value"].iloc[-1]
        ),
        "dividend_terminal_value_float": float(
            dividend_strategy_obj.results["total_value"].iloc[-1]
        ),
        "terminal_value_delta_float": float(
            dividend_strategy_obj.results["total_value"].iloc[-1]
            - baseline_strategy_obj.results["total_value"].iloc[-1]
        ),
        "gross_dividend_cash_float": float(
            getattr(
                dividend_strategy_obj,
                "dividend_cash_gross_total_float",
                0.0,
            )
        ),
        "withholding_cash_float": float(
            getattr(
                dividend_strategy_obj,
                "dividend_withholding_total_float",
                0.0,
            )
        ),
        "net_dividend_cash_float": float(
            getattr(
                dividend_strategy_obj,
                "dividend_cash_net_total_float",
                0.0,
            )
        ),
        "dividend_event_count_int": int(
            len(get_dividend_ledger_df(dividend_strategy_obj))
        ),
        "baseline_transaction_count_int": int(
            len(baseline_strategy_obj.get_transactions())
        ),
        "dividend_transaction_count_int": int(
            len(dividend_strategy_obj.get_transactions())
        ),
        "executed_transaction_skeleton_equal_bool": (
            executed_transaction_skeleton_equal_bool
        ),
        **baseline_cash_diagnostic_dict,
        **dividend_cash_diagnostic_dict,
        "input_and_signal_diagnostics_equal_bool": True,
        "negative_cash_known_gap_bool": negative_cash_breach_bool,
        "negative_cash_policy_status_str": (
            "KNOWN_GAP_REPORTED"
            if negative_cash_breach_bool
            else "NO_NEGATIVE_CASH_OBSERVED"
        ),
        "study_completed_bool": True,
    }

    for output_metric_name_str, summary_metric_name_str in (
        SUMMARY_METRIC_MAP_DICT.items()
    ):
        baseline_metric_float = _summary_metric_float(
            baseline_strategy_obj.summary,
            summary_metric_name_str,
        )
        dividend_metric_float = _summary_metric_float(
            dividend_strategy_obj.summary,
            summary_metric_name_str,
        )
        result_row_dict[f"baseline_{output_metric_name_str}"] = (
            baseline_metric_float
        )
        result_row_dict[f"dividend_{output_metric_name_str}"] = (
            dividend_metric_float
        )
        result_row_dict[f"delta_{output_metric_name_str}"] = (
            dividend_metric_float - baseline_metric_float
        )
    return result_row_dict


def _write_strategy_artifacts(
    strategy_output_path: Path,
    *,
    baseline_strategy_obj: Strategy,
    dividend_strategy_obj: Strategy,
) -> None:
    strategy_output_path.mkdir(parents=True, exist_ok=False)
    equity_and_cash_df = pd.DataFrame(
        {
            "baseline_total_value_float": baseline_strategy_obj.results[
                "total_value"
            ].astype(float),
            "dividend_total_value_float": dividend_strategy_obj.results[
                "total_value"
            ].astype(float),
            "baseline_cash_float": baseline_strategy_obj.results["cash"].astype(
                float
            ),
            "dividend_cash_float": dividend_strategy_obj.results["cash"].astype(
                float
            ),
        }
    )
    equity_and_cash_df.to_csv(strategy_output_path / "equity_and_cash.csv")
    negative_cash_mask_ser = (
        equity_and_cash_df["baseline_cash_float"].lt(-1e-8)
        | equity_and_cash_df["dividend_cash_float"].lt(-1e-8)
    )
    equity_and_cash_df.loc[negative_cash_mask_ser].to_csv(
        strategy_output_path / "negative_cash_days.csv"
    )
    get_dividend_ledger_df(dividend_strategy_obj).to_csv(
        strategy_output_path / "dividend_ledger.csv",
        index=False,
    )
    baseline_strategy_obj.get_transactions().to_csv(
        strategy_output_path / "baseline_transactions.csv",
        index=False,
    )
    dividend_strategy_obj.get_transactions().to_csv(
        strategy_output_path / "dividend_transactions.csv",
        index=False,
    )
    baseline_strategy_obj.summary.to_csv(
        strategy_output_path / "baseline_summary.csv"
    )
    dividend_strategy_obj.summary.to_csv(
        strategy_output_path / "dividend_summary.csv"
    )


def _markdown_value_str(value_obj: object, decimals_int: int = 3) -> str:
    numeric_value_float = float(
        pd.to_numeric(pd.Series([value_obj]), errors="coerce").iloc[0]
    )
    if not np.isfinite(numeric_value_float):
        return ""
    return f"{numeric_value_float:.{decimals_int}f}"


def _write_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    *,
    withholding_rate_float: float,
) -> None:
    negative_cash_known_gap_bool = bool(
        summary_df["negative_cash_known_gap_bool"].astype(bool).any()
    )
    report_line_list = [
        "# WIRED Dividend Cash Ledger A/B",
        "",
        "## Verdict",
        "",
        (
            "**COMPLETED WITH A KNOWN CASH GAP**"
            if negative_cash_known_gap_bool
            else "**COMPLETED — no negative cash observed.**"
        ),
        "",
        "This is a research-only accounting comparison. It does not change the "
        "engine, LIVE, releases, sizing policy, signal definitions, execution "
        "rules, cost rates, or the intentional 0% return on positive cash. "
        "Realized quantities and therefore realized costs may change with NAV.",
        "",
        "The candidate credits Norgate's entitlement-session `Dividend` on the "
        "next market session before open orders, using shares held before that "
        "open. Dividends remain cash; there is no automatic reinvestment.",
        "",
        f"Withholding rate used: `{withholding_rate_float:.2%}`.",
        "",
        "## Results",
        "",
        "| strategy | baseline CAGR % | dividend CAGR % | delta CAGR pp | "
        "baseline Sharpe | dividend Sharpe | net dividends $ | "
        "max negative days (A/B) | max negative days % (A/B) | "
        "max episodes (A/B) | worst cash $ (A/B) | "
        "worst cash % NAV (A/B) | worst average deficit % NAV (A/B) | "
        "cash-gap status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | --- |",
    ]
    for _, result_ser in summary_df.iterrows():
        report_line_list.append(
            "| "
            f"{result_ser['strategy_module_str']} | "
            f"{_markdown_value_str(result_ser['baseline_cagr_pct_float'])} | "
            f"{_markdown_value_str(result_ser['dividend_cagr_pct_float'])} | "
            f"{_markdown_value_str(result_ser['delta_cagr_pct_float'])} | "
            f"{_markdown_value_str(result_ser['baseline_sharpe_float'])} | "
            f"{_markdown_value_str(result_ser['dividend_sharpe_float'])} | "
            f"{_markdown_value_str(result_ser['net_dividend_cash_float'], 2)} | "
            f"{max(int(result_ser['baseline_negative_cash_day_count_int']), int(result_ser['dividend_negative_cash_day_count_int']))} | "
            f"{_markdown_value_str(100.0 * max(float(result_ser['baseline_negative_cash_day_fraction_float']), float(result_ser['dividend_negative_cash_day_fraction_float'])), 2)} | "
            f"{max(int(result_ser['baseline_negative_cash_episode_count_int']), int(result_ser['dividend_negative_cash_episode_count_int']))} | "
            f"{_markdown_value_str(min(float(result_ser['baseline_minimum_cash_float']), float(result_ser['dividend_minimum_cash_float'])), 2)} | "
            f"{_markdown_value_str(100.0 * min(float(result_ser['baseline_minimum_cash_weight_float']), float(result_ser['dividend_minimum_cash_weight_float'])), 3)} | "
            f"{_markdown_value_str(100.0 * min(float(result_ser['baseline_average_negative_cash_weight_float']), float(result_ser['dividend_average_negative_cash_weight_float'])), 3)} | "
            f"{result_ser['negative_cash_policy_status_str']} |"
        )
    report_line_list.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This isolates the accounting ledger only. It does not neutralize "
            "ex-dividend moves inside DV2, QPI, NDX momentum, ATR, or trend signals.",
            "- A later trade can contain a different share quantity because the "
            "candidate account has a different NAV. That is an intended consequence, "
            "not automatic dividend reinvestment.",
            "- `executed_transaction_skeleton_equal_bool` compares executed date, "
            "asset, and direction only. It is a coarse smoke check, not full "
            "execution parity. A mismatch can arise legitimately when a changed NAV "
            "changes target-share rounding; exact inputs and stored signal "
            "diagnostics are hard-gated separately.",
            "- Positive cash earns 0% by intentional owner policy. Negative cash "
            "has no financing charge in the current engine. Per owner decision, it "
            "is reported as a known gap and does not stop this research study.",
            "- The compact cash columns show the conservative worst value selected "
            "independently across baseline and dividend ledgers. `summary.csv` "
            "retains each ledger's separate diagnostics.",
            "- Model A recognizes economic entitlement on ex-date and makes it "
            "spendable cash before the open. Broker cash commonly posts on pay date; "
            "therefore this is not yet live cash-posting parity.",
            "- Current WIRED strategies are long-only. Short manufactured-dividend "
            "debits are unit-tested but not exercised by these A/B runs.",
        ]
    )
    (output_path / "REPORT.md").write_text(
        "\n".join(report_line_list) + "\n",
        encoding="utf-8",
    )


def run_wired_dividend_cash_ledger_study(
    *,
    strategy_module_tuple: tuple[str, ...] = WIRED_STRATEGY_MODULE_TUPLE,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    withholding_rate_float: float = 0.0,
    show_progress_bool: bool = False,
    parity_artifact_path_str: str | None = None,
) -> Path:
    """Run the unchanged and dividend-aware ledgers for each selected module."""
    validated_rate_float = _validated_withholding_rate_float(
        withholding_rate_float
    )
    parity_artifact_path, parity_artifact_sha256_str = (
        _resolve_parity_artifact(
            parity_artifact_path_str,
            output_dir_str=output_dir_str,
        )
    )
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="accounting",
        entity_id_str="wired_dividend_cash_ledger",
        analysis_type_str="ab_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    run_config_dict = {
        "strategy_module_list": list(strategy_module_tuple),
        "backtest_start_date_str": backtest_start_date_str,
        "end_date_str": end_date_str,
        "capital_base_float": float(capital_base_float),
        "withholding_rate_float": validated_rate_float,
        "positive_cash_rate_float": 0.0,
        "dividend_timing_str": (
            "previous_bar entitlement credited before current_bar open"
        ),
        "automatic_reinvestment_bool": False,
        "cash_posting_model_str": (
            "Model A economic ex-date spendable cash; not broker pay-date posting"
        ),
        "parity_artifact_path_str": str(parity_artifact_path),
        "parity_artifact_sha256_str": parity_artifact_sha256_str,
        "exclusive_serial_process_required_bool": True,
        "negative_cash_policy_str": (
            "report_known_gap_without_financing_or_fail_closed_gate"
        ),
    }
    (output_path / "run_config.json").write_text(
        json.dumps(run_config_dict, indent=2),
        encoding="utf-8",
    )

    result_row_dict_list: list[dict[str, object]] = []
    for strategy_index_int, strategy_module_str in enumerate(
        strategy_module_tuple,
        start=1,
    ):
        print(
            f"[{strategy_index_int}/{len(strategy_module_tuple)}] "
            f"{strategy_module_str}: baseline",
            flush=True,
        )
        strategy_module_obj = importlib.import_module(strategy_module_str)
        builder_fn = strategy_module_obj.build_capacity_analysis_inputs
        builder_keyword_dict = _builder_keyword_dict(
            builder_fn,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
            show_progress_bool=show_progress_bool,
        )
        baseline_input_dict = builder_fn(**builder_keyword_dict)
        baseline_strategy_obj = baseline_input_dict["strategy_obj"]

        print(
            f"[{strategy_index_int}/{len(strategy_module_tuple)}] "
            f"{strategy_module_str}: dividend ledger",
            flush=True,
        )
        with research_dividend_cash_ledger_context(
            withholding_rate_float=validated_rate_float
        ):
            dividend_input_dict = builder_fn(**builder_keyword_dict)
        dividend_strategy_obj = dividend_input_dict["strategy_obj"]
        _assert_pair_inputs_equal(
            baseline_input_dict,
            dividend_input_dict,
        )

        result_row_dict = _strategy_result_row_dict(
            strategy_module_str,
            baseline_strategy_obj=baseline_strategy_obj,
            dividend_strategy_obj=dividend_strategy_obj,
        )
        result_row_dict_list.append(result_row_dict)
        strategy_slug_str = strategy_module_str.rsplit(".", maxsplit=1)[-1]
        _write_strategy_artifacts(
            output_path / strategy_slug_str,
            baseline_strategy_obj=baseline_strategy_obj,
            dividend_strategy_obj=dividend_strategy_obj,
        )
        pd.DataFrame(result_row_dict_list).to_csv(
            output_path / "summary.csv",
            index=False,
        )

    summary_df = pd.DataFrame(result_row_dict_list)
    _write_report(
        output_path,
        summary_df,
        withholding_rate_float=validated_rate_float,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run research-only dividend cash accounting A/Bs."
    )
    parser_obj.add_argument(
        "--strategy",
        action="append",
        choices=WIRED_STRATEGY_MODULE_TUPLE,
        dest="strategy_module_list",
        help="Repeat to run a subset. Default: all seven WIRED modules.",
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--start-date")
    parser_obj.add_argument("--end-date")
    parser_obj.add_argument("--capital-base", type=float, default=100_000.0)
    parser_obj.add_argument("--withholding-rate", type=float, default=0.0)
    parser_obj.add_argument(
        "--parity-artifact",
        help="Passed parity-study directory. Default: latest passed artifact.",
    )
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    strategy_module_tuple = (
        WIRED_STRATEGY_MODULE_TUPLE
        if args_obj.strategy_module_list is None
        else tuple(args_obj.strategy_module_list)
    )
    output_path = run_wired_dividend_cash_ledger_study(
        strategy_module_tuple=strategy_module_tuple,
        output_dir_str=args_obj.output_dir,
        backtest_start_date_str=args_obj.start_date,
        end_date_str=args_obj.end_date,
        capital_base_float=args_obj.capital_base,
        withholding_rate_float=args_obj.withholding_rate,
        show_progress_bool=args_obj.show_progress,
        parity_artifact_path_str=args_obj.parity_artifact,
    )
    print(f"Saved study artifacts to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
