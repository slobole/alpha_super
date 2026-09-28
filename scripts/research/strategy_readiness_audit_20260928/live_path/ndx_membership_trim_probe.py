"""
Audit probe (read-only, local Norgate): does the 5-session membership tail trim
(data/norgate_loader.py build_index_constituent_matrix and
scripts/export_norgate_snapshot.py _load_index_constituent_matrix_df) ever drop a
CURRENT Nasdaq-100 member at a live month-end decision date?

Live at month-end M sees Norgate data ending at M. The trim fires for a symbol when
its NONE-padded index_constituent_timeseries does not end on the $SPX last date.
For a member at M this happens exactly when the symbol has no constituent row on M
(for example a trading halt on M, or constituent data lagging price data).

Outputs:
- results/research/strategy_readiness_audit_20260928/live_path/ndx_trim_current_state.csv
- results/research/strategy_readiness_audit_20260928/live_path/ndx_trim_historical_month_end_events.csv
"""
from __future__ import annotations

from pathlib import Path

import norgatedata
import pandas as pd

OUTPUT_DIR_PATH_OBJ = Path("results/research/strategy_readiness_audit_20260928/live_path")
INDEX_NAME_STR = "Nasdaq 100"


def main() -> None:
    OUTPUT_DIR_PATH_OBJ.mkdir(parents=True, exist_ok=True)
    spx_index = norgatedata.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index
    last_trading_day_ts = pd.Timestamp(spx_index[-1])
    session_index = pd.DatetimeIndex(spx_index)
    month_end_session_index = pd.DatetimeIndex(
        pd.Series(session_index, index=session_index.to_period("M")).groupby(level=0).max().to_numpy()
    )
    month_end_session_index = month_end_session_index[
        (month_end_session_index >= pd.Timestamp("2004-01-01"))
        & (month_end_session_index < last_trading_day_ts)
    ]

    symbol_list = norgatedata.watchlist_symbols(f"{INDEX_NAME_STR} Current & Past")
    current_row_list: list[dict[str, object]] = []
    event_row_list: list[dict[str, object]] = []
    for symbol_str in symbol_list:
        constituent_df = norgatedata.index_constituent_timeseries(
            symbol_str, INDEX_NAME_STR, timeseriesformat="pandas-dataframe"
        )
        if constituent_df is None or len(constituent_df) == 0:
            continue
        member_ser = constituent_df["Index Constituent"]
        if member_ser.sum() <= 0:
            continue
        member_row_index = pd.DatetimeIndex(member_ser.index[member_ser == 1])
        all_row_index = pd.DatetimeIndex(member_ser.index)
        last_row_ts = pd.Timestamp(all_row_index[-1])
        last_member_row_ts = pd.Timestamp(member_row_index[-1])
        is_current_member_bool = bool(member_ser.iloc[-1] == 1)
        if is_current_member_bool:
            current_row_list.append(
                {
                    "symbol_str": symbol_str,
                    "last_constituent_row_date_str": last_row_ts.date().isoformat(),
                    "spx_last_date_str": last_trading_day_ts.date().isoformat(),
                    "trim_fires_today_bool": bool(last_member_row_ts != last_trading_day_ts),
                }
            )
        # Historical: member at month-end M (member on the last constituent row on or
        # before M, and still a member on the next constituent row after M) but with no
        # constituent row ON M -> live at M would have trimmed/zeroed it.
        member_ffill_ser = member_ser.reindex(session_index).ffill()
        member_bfill_ser = member_ser.reindex(session_index).bfill()
        for month_end_ts in month_end_session_index:
            if month_end_ts in all_row_index:
                continue
            if month_end_ts < all_row_index[0] or month_end_ts > all_row_index[-1]:
                continue
            if member_ffill_ser.get(month_end_ts) == 1 and member_bfill_ser.get(month_end_ts) == 1:
                previous_row_ts = all_row_index[all_row_index < month_end_ts][-1]
                event_row_list.append(
                    {
                        "symbol_str": symbol_str,
                        "month_end_date_str": month_end_ts.date().isoformat(),
                        "previous_constituent_row_date_str": previous_row_ts.date().isoformat(),
                    }
                )

    current_df = pd.DataFrame(current_row_list)
    event_df = pd.DataFrame(event_row_list, columns=["symbol_str", "month_end_date_str", "previous_constituent_row_date_str"])
    current_df.to_csv(OUTPUT_DIR_PATH_OBJ / "ndx_trim_current_state.csv", index=False)
    event_df.to_csv(OUTPUT_DIR_PATH_OBJ / "ndx_trim_historical_month_end_events.csv", index=False)
    print(f"spx_last_date={last_trading_day_ts.date()}")
    print(f"current_members={len(current_df)} trim_fires_today={int(current_df['trim_fires_today_bool'].sum())}")
    print(current_df.loc[current_df["trim_fires_today_bool"]].to_string())
    print(f"historical member-without-row-at-month-end events since 2004: {len(event_df)}")
    print(event_df.to_string())


if __name__ == "__main__":
    main()
