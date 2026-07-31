"""
Liquidity / participation diagnostic for corr-penalty selection audits.

Reads the per-month selection audits written by
run_atr_normalized_ndx_corr_penalty_sweep, loads dollar-volume history for
every symbol that was ever selected, and reports how large each monthly
rebalance order is relative to the name's trailing dollar ADV.

Formulas (per selected symbol i at decision date t):

    adv_usd_{i,t}
        = median(Volume_{i,d} * UnadjustedClose_{i,d} for d in the 20
          trading days ending at t)                      *** trailing only ***

    position_usd_t = equity_t / max_positions
    participation_{i,t} = position_usd_t / adv_usd_{i,t}

Reported at the backtest capital base and rescaled to a hypothetical AUM so
the capacity intuition transfers. This is a coarse screen, not a TCA — the
official capacity tool remains CapacityAnalysis (see G-012).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from data.norgate_loader import load_raw_prices


def run_liquidity_check(
    sweep_output_dir_str: str,
    audit_variant_label_str: str,
    equity_column_str: str,
    max_positions_int: int,
    hypothetical_aum_usd_float: float = 1_000_000.0,
    adv_window_int: int = 20,
) -> pd.DataFrame:
    sweep_output_path = Path(sweep_output_dir_str)
    audit_df = pd.read_csv(
        sweep_output_path / f"selection_audit_{audit_variant_label_str}.csv",
        index_col=0,
        parse_dates=True,
    )
    equity_df = pd.read_csv(sweep_output_path / "equity_curve.csv", index_col=0, parse_dates=True)
    equity_ser = equity_df[equity_column_str].dropna().astype(float)

    selected_symbol_set: set[str] = set()
    selection_map: dict[pd.Timestamp, list[str]] = {}
    for decision_date_ts, row_ser in audit_df.iterrows():
        symbol_list_str = str(row_ser["selected_symbol_list"])
        symbol_list = [s for s in symbol_list_str.split("|") if s]
        selection_map[pd.Timestamp(decision_date_ts)] = symbol_list
        selected_symbol_set.update(symbol_list)

    if len(selected_symbol_set) == 0:
        raise RuntimeError("No selected symbols found in the audit file.")

    first_decision_ts = min(selection_map)
    price_df = load_raw_prices(
        symbols=sorted(selected_symbol_set),
        benchmarks=[],
        start_date=(first_decision_ts - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
        end_date=None,
    )

    dollar_volume_frame_map: dict[str, pd.Series] = {}
    for symbol_str in sorted(selected_symbol_set):
        volume_key = (symbol_str, "Volume")
        unadjusted_close_key = (symbol_str, "Unadjusted Close")
        if volume_key not in price_df.columns or unadjusted_close_key not in price_df.columns:
            continue
        # *** CRITICAL*** ADV uses only data on or before the decision date via
        # the rolling window; unadjusted close keeps the dollar value in
        # as-traded terms across splits.
        dollar_volume_ser = (
            price_df[volume_key].astype(float) * price_df[unadjusted_close_key].astype(float)
        )
        dollar_volume_frame_map[symbol_str] = dollar_volume_ser.rolling(
            window=adv_window_int, min_periods=max(5, adv_window_int // 2)
        ).median()

    participation_row_list: list[dict[str, object]] = []
    for decision_date_ts, symbol_list in sorted(selection_map.items()):
        equity_asof_ser = equity_ser.loc[:decision_date_ts]
        if len(equity_asof_ser) == 0:
            continue
        position_usd_float = float(equity_asof_ser.iloc[-1]) / float(max_positions_int)
        for symbol_str in symbol_list:
            adv_ser = dollar_volume_frame_map.get(symbol_str)
            adv_asof_ser = adv_ser.loc[:decision_date_ts].dropna() if adv_ser is not None else None
            adv_usd_float = float(adv_asof_ser.iloc[-1]) if adv_asof_ser is not None and len(adv_asof_ser) > 0 else np.nan
            participation_row_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "adv_usd_float": adv_usd_float,
                    "position_usd_float": position_usd_float,
                    "participation_float": (
                        position_usd_float / adv_usd_float if np.isfinite(adv_usd_float) and adv_usd_float > 0 else np.nan
                    ),
                }
            )

    participation_df = pd.DataFrame(participation_row_list)
    valid_df = participation_df.dropna(subset=["participation_float"])
    missing_count_int = int(len(participation_df) - len(valid_df))

    backtest_equity_start_float = float(equity_ser.iloc[0])
    scale_float = hypothetical_aum_usd_float / backtest_equity_start_float

    print(f"variant: {audit_variant_label_str}  name-months: {len(participation_df)}  missing ADV: {missing_count_int}")
    print(f"ADV$ of selected names: median {valid_df['adv_usd_float'].median():,.0f}, "
          f"p5 {valid_df['adv_usd_float'].quantile(0.05):,.0f}, min {valid_df['adv_usd_float'].min():,.0f}")
    print("participation (order $ / ADV$), at backtest capital "
          f"({backtest_equity_start_float:,.0f} start):")
    print(f"  median {valid_df['participation_float'].median():.5%}  "
          f"p95 {valid_df['participation_float'].quantile(0.95):.5%}  "
          f"max {valid_df['participation_float'].max():.5%}")
    print(f"scaled to AUM {hypothetical_aum_usd_float:,.0f} (linear approximation):")
    print(f"  median {valid_df['participation_float'].median() * scale_float:.4%}  "
          f"p95 {valid_df['participation_float'].quantile(0.95) * scale_float:.4%}  "
          f"max {valid_df['participation_float'].max() * scale_float:.4%}")
    worst_df = valid_df.nlargest(5, "participation_float")[
        ["decision_date_ts", "symbol_str", "adv_usd_float", "participation_float"]
    ]
    print("worst 5 name-months by participation:")
    print(worst_df.to_string(index=False))

    out_path = sweep_output_path / f"liquidity_check_{audit_variant_label_str}.csv"
    participation_df.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")
    return participation_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--variant", required=True, help="audit variant label, e.g. n20_lam1_w126")
    parser.add_argument("--equity-column", required=True, help="equity_curve.csv column name")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--aum", type=float, default=1_000_000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_liquidity_check(
        sweep_output_dir_str=args.sweep_dir,
        audit_variant_label_str=args.variant,
        equity_column_str=args.equity_column,
        max_positions_int=args.max_positions,
        hypothetical_aum_usd_float=args.aum,
    )
