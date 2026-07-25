"""Research-only parity check for explicit dividends versus Norgate total return."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from data.norgate_loader import load_price_timeseries
from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
)


DEFAULT_SYMBOL_TUPLE = (
    "AAPL",
    "MSFT",
    "XOM",
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "UUP",
    "DBC",
    "BTAL",
    "TQQQ",
)
DEFAULT_START_DATE_STR = "2000-01-01"
MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT = 0.02
EVENT_MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT = 2.0
MAX_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT = 10.0
TERMINAL_WEALTH_ERROR_LIMIT_BPS_FLOAT = 20.0


def build_dividend_parity_frame_df(
    capital_price_df: pd.DataFrame,
    total_return_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align price/dividend data and compute the two timing hypotheses."""
    required_capital_field_set = {"Close", "Dividend"}
    missing_capital_field_set = required_capital_field_set.difference(
        capital_price_df.columns
    )
    if missing_capital_field_set:
        raise ValueError(
            "capital_price_df is missing fields: "
            f"{sorted(missing_capital_field_set)}"
        )
    if "Close" not in total_return_price_df.columns:
        raise ValueError("total_return_price_df is missing field: Close")

    if not capital_price_df.index.is_unique:
        raise ValueError("capital_price_df index contains duplicate dates.")
    if not total_return_price_df.index.is_unique:
        raise ValueError("total_return_price_df index contains duplicate dates.")
    if not capital_price_df.index.equals(total_return_price_df.index):
        capital_only_date_idx = capital_price_df.index.difference(
            total_return_price_df.index
        )
        total_return_only_date_idx = total_return_price_df.index.difference(
            capital_price_df.index
        )
        raise ValueError(
            "CAPITALSPECIAL and TOTALRETURN calendars differ: "
            f"capital_only_count={len(capital_only_date_idx)} "
            f"total_return_only_count={len(total_return_only_date_idx)}"
        )
    common_date_idx = capital_price_df.index.sort_values()
    if len(common_date_idx) < 2:
        raise ValueError("At least two common price observations are required.")

    capital_close_ser = pd.to_numeric(
        capital_price_df.loc[common_date_idx, "Close"],
        errors="coerce",
    ).astype(float)
    dividend_entitlement_ser = pd.to_numeric(
        capital_price_df.loc[common_date_idx, "Dividend"],
        errors="coerce",
    ).astype(float)
    total_return_close_ser = pd.to_numeric(
        total_return_price_df.loc[common_date_idx, "Close"],
        errors="coerce",
    ).astype(float)

    invalid_price_mask_ser = (
        ~np.isfinite(capital_close_ser)
        | ~np.isfinite(total_return_close_ser)
        | capital_close_ser.le(0.0)
        | total_return_close_ser.le(0.0)
    )
    if invalid_price_mask_ser.any():
        invalid_date_str = pd.Timestamp(
            common_date_idx[invalid_price_mask_ser.to_numpy()][0]
        ).date().isoformat()
        raise ValueError(f"Invalid close price on {invalid_date_str}.")
    if not np.isfinite(dividend_entitlement_ser).all():
        raise ValueError("Dividend contains a non-finite value.")

    # *** CRITICAL*** Norgate stamps Dividend on entitlement session T.
    # Economic return recognizes it on the next market session T+1, when the
    # ex-dividend price change occurs. A same-session use is one day early.
    dividend_ex_date_ser = dividend_entitlement_ser.shift(1).fillna(0.0)
    previous_capital_close_ser = capital_close_ser.shift(1)
    modeled_total_return_ser = (
        (capital_close_ser + dividend_ex_date_ser)
        / previous_capital_close_ser
        - 1.0
    )
    same_session_placebo_return_ser = (
        (capital_close_ser + dividend_entitlement_ser)
        / previous_capital_close_ser
        - 1.0
    )
    norgate_total_return_ser = total_return_close_ser.pct_change(
        fill_method=None
    )

    parity_frame_df = pd.DataFrame(
        {
            "capital_close_float": capital_close_ser,
            "dividend_entitlement_float": dividend_entitlement_ser,
            "dividend_ex_date_float": dividend_ex_date_ser,
            "total_return_close_float": total_return_close_ser,
            "modeled_total_return_float": modeled_total_return_ser,
            "same_session_placebo_return_float": same_session_placebo_return_ser,
            "norgate_total_return_float": norgate_total_return_ser,
        },
        index=common_date_idx,
    )
    parity_frame_df.index.name = "date"
    parity_frame_df["modeled_error_bps_float"] = (
        parity_frame_df["modeled_total_return_float"]
        - parity_frame_df["norgate_total_return_float"]
    ) * 10_000.0
    parity_frame_df["same_session_placebo_error_bps_float"] = (
        parity_frame_df["same_session_placebo_return_float"]
        - parity_frame_df["norgate_total_return_float"]
    ) * 10_000.0
    return parity_frame_df


def compute_symbol_summary_dict(
    symbol_str: str,
    parity_frame_df: pd.DataFrame,
) -> dict[str, object]:
    """Summarize daily and terminal wealth parity for one security."""
    valid_parity_df = parity_frame_df.dropna(
        subset=[
            "modeled_total_return_float",
            "same_session_placebo_return_float",
            "norgate_total_return_float",
        ]
    )
    if len(valid_parity_df) == 0:
        raise ValueError(f"No valid return observations for {symbol_str}.")

    modeled_wealth_ser = (
        1.0 + valid_parity_df["modeled_total_return_float"]
    ).cumprod()
    norgate_wealth_ser = (
        1.0 + valid_parity_df["norgate_total_return_float"]
    ).cumprod()
    placebo_wealth_ser = (
        1.0 + valid_parity_df["same_session_placebo_return_float"]
    ).cumprod()

    modeled_error_bps_ser = valid_parity_df["modeled_error_bps_float"].abs()
    dividend_event_mask_ser = valid_parity_df["dividend_ex_date_float"].ne(0.0)
    event_error_bps_ser = modeled_error_bps_ser.loc[dividend_event_mask_ser]
    event_mean_absolute_error_bps_float = (
        0.0 if len(event_error_bps_ser) == 0 else float(event_error_bps_ser.mean())
    )
    placebo_error_bps_ser = valid_parity_df[
        "same_session_placebo_error_bps_float"
    ].abs()
    terminal_wealth_error_bps_float = (
        float(modeled_wealth_ser.iloc[-1] / norgate_wealth_ser.iloc[-1]) - 1.0
    ) * 10_000.0
    placebo_terminal_wealth_error_bps_float = (
        float(placebo_wealth_ser.iloc[-1] / norgate_wealth_ser.iloc[-1]) - 1.0
    ) * 10_000.0
    initial_capital_close_float = float(
        parity_frame_df["capital_close_float"].iloc[0]
    )
    dividend_cash_per_initial_share_float = float(
        valid_parity_df["dividend_ex_date_float"].sum()
    )
    fixed_share_cash_terminal_wealth_float = (
        float(valid_parity_df["capital_close_float"].iloc[-1])
        + dividend_cash_per_initial_share_float
    ) / initial_capital_close_float
    fixed_share_cash_vs_total_return_error_bps_float = (
        fixed_share_cash_terminal_wealth_float
        / float(norgate_wealth_ser.iloc[-1])
        - 1.0
    ) * 10_000.0

    mean_absolute_error_bps_float = float(modeled_error_bps_ser.mean())
    max_absolute_error_bps_float = float(modeled_error_bps_ser.max())
    pass_bool = (
        mean_absolute_error_bps_float
        <= MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        and event_mean_absolute_error_bps_float
        <= EVENT_MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        and max_absolute_error_bps_float
        <= MAX_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        and abs(terminal_wealth_error_bps_float)
        <= TERMINAL_WEALTH_ERROR_LIMIT_BPS_FLOAT
    )

    return {
        "symbol_str": symbol_str,
        "start_date_str": pd.Timestamp(valid_parity_df.index[0]).date().isoformat(),
        "end_date_str": pd.Timestamp(valid_parity_df.index[-1]).date().isoformat(),
        "observation_count_int": int(len(valid_parity_df)),
        "dividend_event_count_int": int(
            parity_frame_df["dividend_entitlement_float"].ne(0.0).sum()
        ),
        "mean_absolute_error_bps_float": mean_absolute_error_bps_float,
        "event_mean_absolute_error_bps_float": (
            event_mean_absolute_error_bps_float
        ),
        "max_absolute_error_bps_float": max_absolute_error_bps_float,
        "terminal_wealth_error_bps_float": terminal_wealth_error_bps_float,
        "same_session_placebo_mean_absolute_error_bps_float": float(
            placebo_error_bps_ser.mean()
        ),
        "same_session_placebo_max_absolute_error_bps_float": float(
            placebo_error_bps_ser.max()
        ),
        "same_session_placebo_terminal_wealth_error_bps_float": (
            placebo_terminal_wealth_error_bps_float
        ),
        "dividend_cash_per_initial_share_float": (
            dividend_cash_per_initial_share_float
        ),
        "fixed_share_cash_terminal_wealth_float": (
            fixed_share_cash_terminal_wealth_float
        ),
        "fixed_share_cash_vs_total_return_error_bps_float": (
            fixed_share_cash_vs_total_return_error_bps_float
        ),
        "modeled_terminal_wealth_float": float(modeled_wealth_ser.iloc[-1]),
        "norgate_terminal_wealth_float": float(norgate_wealth_ser.iloc[-1]),
        "pass_bool": bool(pass_bool),
    }


def build_dividend_event_df(
    symbol_str: str,
    parity_frame_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one audit row per shifted ex-dividend event."""
    event_date_idx = parity_frame_df.index[
        parity_frame_df["dividend_ex_date_float"].ne(0.0)
    ]
    event_row_dict_list: list[dict[str, object]] = []
    for event_date_ts in event_date_idx:
        event_position_int = int(parity_frame_df.index.get_loc(event_date_ts))
        if event_position_int <= 0:
            continue
        entitlement_date_ts = parity_frame_df.index[event_position_int - 1]
        event_row_dict_list.append(
            {
                "symbol_str": symbol_str,
                "entitlement_date_str": pd.Timestamp(
                    entitlement_date_ts
                ).date().isoformat(),
                "ex_date_str": pd.Timestamp(event_date_ts).date().isoformat(),
                "dividend_per_share_float": float(
                    parity_frame_df.loc[
                        event_date_ts,
                        "dividend_ex_date_float",
                    ]
                ),
                "previous_capital_close_float": float(
                    parity_frame_df.iloc[event_position_int - 1][
                        "capital_close_float"
                    ]
                ),
                "ex_date_capital_close_float": float(
                    parity_frame_df.loc[event_date_ts, "capital_close_float"]
                ),
                "modeled_total_return_float": float(
                    parity_frame_df.loc[
                        event_date_ts,
                        "modeled_total_return_float",
                    ]
                ),
                "norgate_total_return_float": float(
                    parity_frame_df.loc[
                        event_date_ts,
                        "norgate_total_return_float",
                    ]
                ),
                "modeled_error_bps_float": float(
                    parity_frame_df.loc[
                        event_date_ts,
                        "modeled_error_bps_float",
                    ]
                ),
                "same_session_placebo_error_bps_float": float(
                    parity_frame_df.loc[
                        event_date_ts,
                        "same_session_placebo_error_bps_float",
                    ]
                ),
            }
        )
    return pd.DataFrame(event_row_dict_list)


def _markdown_table_str(summary_df: pd.DataFrame) -> str:
    column_tuple = (
        "symbol_str",
        "dividend_event_count_int",
        "event_mean_absolute_error_bps_float",
        "max_absolute_error_bps_float",
        "terminal_wealth_error_bps_float",
        "fixed_share_cash_vs_total_return_error_bps_float",
        "same_session_placebo_max_absolute_error_bps_float",
        "pass_bool",
    )
    header_str = (
        "| Symbol | Dividends | Event mean error (bps) | Max abs error (bps) | "
        "Reinvested parity error (bps) | Cash-only vs TR (bps) | "
        "Wrong-timing max error (bps) | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---:|:---:|"
    )
    row_str_list = [header_str]
    for row_obj in summary_df.loc[:, column_tuple].itertuples(index=False):
        row_str_list.append(
            f"| {row_obj.symbol_str} | {row_obj.dividend_event_count_int} | "
            f"{row_obj.event_mean_absolute_error_bps_float:.6f} | "
            f"{row_obj.max_absolute_error_bps_float:.6f} | "
            f"{row_obj.terminal_wealth_error_bps_float:.6f} | "
            f"{row_obj.fixed_share_cash_vs_total_return_error_bps_float:.6f} | "
            f"{row_obj.same_session_placebo_max_absolute_error_bps_float:.6f} | "
            f"{'PASS' if row_obj.pass_bool else 'FAIL'} |"
        )
    return "\n".join(row_str_list)


def run_dividend_parity_study(
    symbol_tuple: tuple[str, ...] = DEFAULT_SYMBOL_TUPLE,
    start_date_str: str = DEFAULT_START_DATE_STR,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
) -> Path:
    """Run the direct-Norgate parity study and save its audit artifacts."""
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="accounting",
        entity_id_str="dividend_total_return_parity",
        analysis_type_str="parity_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    summary_row_dict_list: list[dict[str, object]] = []
    event_df_list: list[pd.DataFrame] = []
    for symbol_str in symbol_tuple:
        capital_price_df = load_price_timeseries(
            symbol_str,
            adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        total_return_price_df = load_price_timeseries(
            symbol_str,
            adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        parity_frame_df = build_dividend_parity_frame_df(
            capital_price_df=capital_price_df,
            total_return_price_df=total_return_price_df,
        )
        summary_row_dict_list.append(
            compute_symbol_summary_dict(
                symbol_str=symbol_str,
                parity_frame_df=parity_frame_df,
            )
        )
        event_df_list.append(
            build_dividend_event_df(
                symbol_str=symbol_str,
                parity_frame_df=parity_frame_df,
            )
        )

    summary_df = pd.DataFrame(summary_row_dict_list)
    event_df = pd.concat(event_df_list, ignore_index=True)
    overall_pass_bool = bool(summary_df["pass_bool"].all())
    summary_df.to_csv(output_path / "symbol_summary.csv", index=False)
    event_df.to_csv(output_path / "dividend_events.csv", index=False)

    verdict_md_str = "\n".join(
        [
            "# Dividend Total-Return Parity Verdict",
            "",
            f"**Verdict:** {'PASS' if overall_pass_bool else 'FAIL'}",
            "",
            "Research-only validation. No strategy signals, engine ledger, live code, "
            "VPS configuration, costs, or cash-rate policy changed.",
            "",
            "Formula:",
            "",
            "`r_model(T+1) = (Close_CS(T+1) + Dividend_Norgate(T)) / Close_CS(T) - 1`",
            "",
            _markdown_table_str(summary_df),
            "",
            "Acceptance limits:",
            "",
            f"- mean absolute daily error <= {MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT:.2f} bps;",
            f"- mean absolute dividend-event error <= {EVENT_MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT:.2f} bps;",
            f"- maximum absolute daily error <= {MAX_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT:.2f} bps;",
            f"- absolute terminal wealth error <= {TERMINAL_WEALTH_ERROR_LIMIT_BPS_FLOAT:.2f} bps.",
            "",
            "The same-session placebo intentionally posts the dividend one session "
            "too early. Its event-level error is retained as a timing diagnostic, not "
            "as a candidate accounting policy.",
            "",
            "Scope: PASS validates dividend amount/timing by compounding the modeled "
            "economic return, which implicitly reinvests distributions like Norgate "
            "TOTALRETURN. `Cash-only vs TR` instead holds one share and leaves all "
            "dividends in 0%-return cash. It can lead or lag TR depending on the "
            "asset's subsequent return and is not part of the parity acceptance gate.",
        ]
    )
    (output_path / "verdict.md").write_text(verdict_md_str, encoding="utf-8")

    metadata_dict = {
        "study_id_str": "dividend_total_return_parity_v1",
        "research_only_bool": True,
        "symbol_list": list(symbol_tuple),
        "search_space_count_int": 2,
        "tolerance_calibration_str": (
            "Engineering tolerances were frozen before the saved full study "
            "after one local exploratory precheck; this is accounting "
            "validation, not an alpha parameter search."
        ),
        "hypothesis_list": [
            "shift_entitlement_dividend_to_next_session",
            "same_session_placebo",
        ],
        "capital_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
        "reference_adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
        "start_date_str": start_date_str,
        "requested_end_date_str": end_date_str,
        "withholding_rate_float": 0.0,
        "positive_cash_rate_float": 0.0,
        "mean_absolute_error_limit_bps_float": (
            MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        ),
        "event_mean_absolute_error_limit_bps_float": (
            EVENT_MEAN_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        ),
        "max_absolute_error_limit_bps_float": (
            MAX_ABSOLUTE_ERROR_LIMIT_BPS_FLOAT
        ),
        "terminal_wealth_error_limit_bps_float": (
            TERMINAL_WEALTH_ERROR_LIMIT_BPS_FLOAT
        ),
        "overall_pass_bool": overall_pass_bool,
        "pass_scope_str": (
            "Dividend amount and ex-date timing parity under reinvested "
            "economic returns; not non-reinvesting engine-ledger parity."
        ),
        "output_path_str": str(output_path.resolve()),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser_obj = argparse.ArgumentParser(
        description=(
            "Validate CAPITALSPECIAL plus explicit dividends against Norgate "
            "TOTALRETURN without changing the engine."
        )
    )
    parser_obj.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOL_TUPLE),
    )
    parser_obj.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE_STR,
    )
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--output-dir", default="results")
    args_obj = parser_obj.parse_args()

    output_path = run_dividend_parity_study(
        symbol_tuple=tuple(str(symbol_str) for symbol_str in args_obj.symbols),
        start_date_str=str(args_obj.start_date),
        end_date_str=(
            None if args_obj.end_date is None else str(args_obj.end_date)
        ),
        output_dir_str=str(args_obj.output_dir),
    )
    print(f"Saved dividend parity study to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
