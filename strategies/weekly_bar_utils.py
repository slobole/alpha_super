from __future__ import annotations

import pandas as pd


def get_completed_week_decision_date_index(price_date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the actual last available trading date for each completed W-FRI week.

    A trailing partial week is dropped unless the final available date is Friday.
    Holiday-short weeks are retained once a later week is present in the data.
    """
    if len(price_date_index) == 0:
        raise ValueError("price_date_index must not be empty.")

    sorted_date_index = pd.DatetimeIndex(price_date_index).sort_values()
    week_period_idx = sorted_date_index.to_period("W-FRI")
    decision_date_ser = pd.Series(sorted_date_index, index=week_period_idx).groupby(level=0).max()

    last_available_ts = pd.Timestamp(sorted_date_index[-1])
    last_week_end_ts = pd.Timestamp(last_available_ts.to_period("W-FRI").end_time).normalize()
    # *** CRITICAL *** completeness-sensitive: only the final available week
    # can be an in-progress week, so only that last weekly decision is dropped.
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and last_available_ts.normalize() != last_week_end_ts
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    return pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")


def build_completed_week_ohlcv_df(pricing_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily OHLCV data to completed weekly decision bars.

    Weekly bar formulas for symbol i and completed week W:

        Open_W  = first available Open in W
        High_W  = max High in W
        Low_W   = min Low in W
        Close_W = last available Close in W
        Turnover_W = sum Turnover in W
    """
    if len(pricing_data_df.index) == 0:
        raise ValueError("pricing_data_df must not be empty.")
    if not isinstance(pricing_data_df.columns, pd.MultiIndex):
        raise ValueError("pricing_data_df must have MultiIndex columns.")

    sorted_pricing_data_df = pricing_data_df.sort_index()
    decision_date_index = get_completed_week_decision_date_index(sorted_pricing_data_df.index)
    if len(decision_date_index) == 0:
        return pd.DataFrame(columns=pricing_data_df.columns, index=decision_date_index)

    week_period_idx = sorted_pricing_data_df.index.to_period("W-FRI")
    decision_period_idx = decision_date_index.to_period("W-FRI")
    decision_date_by_period_ser = pd.Series(decision_date_index, index=decision_period_idx)

    field_aggregation_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Turnover": "sum",
    }
    aggregated_field_df_list: list[pd.DataFrame] = []

    for field_str, aggregation_str in field_aggregation_map.items():
        field_column_list = [
            column_obj
            for column_obj in sorted_pricing_data_df.columns
            if len(column_obj) >= 2 and str(column_obj[1]) == field_str
        ]
        if len(field_column_list) == 0:
            continue

        field_value_df = sorted_pricing_data_df.loc[:, field_column_list]
        grouped_field_obj = field_value_df.groupby(week_period_idx)
        if aggregation_str == "first":
            aggregated_field_df = grouped_field_obj.first()
        elif aggregation_str == "last":
            aggregated_field_df = grouped_field_obj.last()
        elif aggregation_str == "max":
            aggregated_field_df = grouped_field_obj.max()
        elif aggregation_str == "min":
            aggregated_field_df = grouped_field_obj.min()
        elif aggregation_str == "sum":
            aggregated_field_df = grouped_field_obj.sum(min_count=1)
        else:  # pragma: no cover - defensive branch for the closed map above.
            raise RuntimeError(f"Unsupported weekly aggregation '{aggregation_str}'.")

        aggregated_field_df = aggregated_field_df.reindex(decision_period_idx)
        aggregated_field_df.index = pd.DatetimeIndex(
            decision_date_by_period_ser.reindex(decision_period_idx).to_numpy(),
            name="decision_date_ts",
        )
        aggregated_field_df_list.append(aggregated_field_df)

    if len(aggregated_field_df_list) == 0:
        return pd.DataFrame(index=decision_date_index)

    weekly_bar_df = pd.concat(aggregated_field_df_list, axis=1)
    weekly_bar_df = weekly_bar_df.reindex(columns=pricing_data_df.columns.intersection(weekly_bar_df.columns))
    return weekly_bar_df
