import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import talib


TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.bench import catalog
from alpha.engine.order import MarketOrder
from strategies.hpi.stateful_long import (
    ADV_63_FIELD_STR,
    ENTRY_HORIZON_VOTE_STR,
    HPI_2D_FIELD_STR,
    HPI_5D_FIELD_STR,
    LIQUIDITY_RELATIVE_STR,
    NATR_FIELD_STR,
    RAW_PRICE_FIELD_STR,
    RETURN_2D_FIELD_STR,
    RETURN_5D_FIELD_STR,
    TURNOVER_FIELD_STR,
    HPIStatefulLongStrategy,
    compute_strict_hpi,
    load_exact_hpi_inputs,
    run_hpi_variant,
)


def make_strategy(
    ranking_field_str: str = TURNOVER_FIELD_STR,
    max_positions_int: int = 2,
    entry_mode_str: str = "baseline",
    liquidity_mode_str: str = "none",
    backtest_start_date_str: str | None = None,
) -> HPIStatefulLongStrategy:
    return HPIStatefulLongStrategy(
        name="HPIStatefulLongTest",
        benchmarks=[],
        ranking_field_str=ranking_field_str,
        capital_base=100_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
        max_positions_int=max_positions_int,
        entry_mode_str=entry_mode_str,
        liquidity_mode_str=liquidity_mode_str,
        backtest_start_date_str=backtest_start_date_str,
    )


def make_close_row_ser(
    row_value_dict: dict[tuple[str, str], float],
) -> pd.Series:
    close_row_ser = pd.Series(row_value_dict, dtype=float)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_strategy_declares_total_return_benchmark_provenance():
    strategy_obj = HPIStatefulLongStrategy(
        name="HPIBenchmarkProvenanceTest",
        benchmarks=["$SPX"],
        ranking_field_str=TURNOVER_FIELD_STR,
    )

    assert strategy_obj._performance_benchmark_adjustment_str == "TOTALRETURN"
    assert strategy_obj._data_adjustment_policy_dict == {
        "stock_signal_adjustment_str": "CAPITALSPECIAL",
        "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
        "performance_benchmark_adjustment_str": "TOTALRETURN",
    }


def eligible_field_dict(
    symbol_str: str,
    *,
    turnover_float: float,
    natr_float: float,
) -> dict[tuple[str, str], float]:
    return {
        (symbol_str, "Close"): 105.0,
        (symbol_str, TURNOVER_FIELD_STR): turnover_float,
        (symbol_str, NATR_FIELD_STR): natr_float,
        (symbol_str, "return_3d_ser"): -0.03,
        (symbol_str, "hpi_value_ser"): 20.0,
        (symbol_str, "sma_200_price_ser"): 100.0,
        (symbol_str, "ibs_value_ser"): 0.05,
        (symbol_str, "rsi2_value_ser"): 20.0,
    }


def test_compute_strict_hpi_excludes_current_observation_and_respects_ties():
    nonpositive_return_ser = pd.Series([-0.10, -0.05, -0.05, 0.10, -0.05])
    positive_return_ser = pd.Series([-0.10, 0.02, 0.05, 0.05, 0.05])

    nonpositive_hpi_ser = compute_strict_hpi(
        nonpositive_return_ser,
        lookback_int=4,
    )
    positive_hpi_ser = compute_strict_hpi(
        positive_return_ser,
        lookback_int=4,
    )

    assert nonpositive_hpi_ser.iloc[:4].isna().all()
    assert nonpositive_hpi_ser.iloc[-1] == pytest.approx(100.0)
    assert positive_hpi_ser.iloc[-1] == pytest.approx(0.0)


def test_compute_signals_matches_supplied_hpi_and_natr_formulas():
    date_index = pd.bdate_range("2018-01-02", periods=1_264)
    step_vec = np.arange(len(date_index), dtype=float)
    close_vec = 100.0 + 0.03 * step_vec + 2.0 * np.sin(step_vec * 0.07)
    high_vec = close_vec + 1.0
    low_vec = close_vec - 1.0
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): close_vec - 0.2,
            ("AAA", "High"): high_vec,
            ("AAA", "Low"): low_vec,
            ("AAA", "Close"): close_vec,
            ("AAA", TURNOVER_FIELD_STR): 25_000_000.0 + step_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = make_strategy(ranking_field_str=NATR_FIELD_STR)

    signal_data_df = strategy_obj.compute_signals(pricing_data_df)

    close_price_ser = pricing_data_df[("AAA", "Close")].astype(float)
    # *** CRITICAL*** direct test oracle uses Close_T and Close_(T-3), then
    # compares T only with the 1,260 completed Return3D observations before T.
    return_3d_ser = close_price_ser / close_price_ser.shift(3) - 1.0
    current_return_float = float(return_3d_ser.iloc[-1])
    prior_return_ser = return_3d_ser.iloc[-1_261:-1]
    assert len(prior_return_ser) == 1_260
    assert prior_return_ser.notna().all()
    if current_return_float <= 0.0:
        expected_hpi_float = (
            100.0
            * float(prior_return_ser.le(current_return_float).sum())
            / float(prior_return_ser.le(0.0).sum())
        )
    else:
        expected_hpi_float = (
            100.0
            * float(prior_return_ser.gt(current_return_float).sum())
            / float(prior_return_ser.gt(0.0).sum())
        )

    expected_natr_float = float(
        talib.NATR(
            high_vec,
            low_vec,
            close_vec,
            timeperiod=14,
        )[-1]
    )
    last_date_ts = date_index[-1]
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", "return_3d_ser"),
    ] == pytest.approx(current_return_float)
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", "hpi_value_ser"),
    ] == pytest.approx(expected_hpi_float)
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", NATR_FIELD_STR),
    ] == pytest.approx(expected_natr_float)


def test_compute_signals_adds_vote_and_relative_liquidity_features():
    date_index = pd.bdate_range("2018-01-02", periods=1_264)
    step_vec = np.arange(len(date_index), dtype=float)
    close_vec = 100.0 + 0.03 * step_vec + 2.0 * np.sin(step_vec * 0.07)
    volume_vec = 1_000_000.0 + step_vec
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): close_vec - 0.2,
            ("AAA", "High"): close_vec + 1.0,
            ("AAA", "Low"): close_vec - 1.0,
            ("AAA", "Close"): close_vec,
            ("AAA", "Unadjusted Close"): close_vec,
            ("AAA", "Volume"): volume_vec,
            ("AAA", TURNOVER_FIELD_STR): close_vec * volume_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = make_strategy(
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    )

    signal_data_df = strategy_obj.compute_signals(pricing_data_df)

    assert {
        RETURN_2D_FIELD_STR,
        RETURN_5D_FIELD_STR,
        HPI_2D_FIELD_STR,
        HPI_5D_FIELD_STR,
        RAW_PRICE_FIELD_STR,
        ADV_63_FIELD_STR,
    }.issubset(signal_data_df["AAA"].columns)
    expected_adv_float = float(
        pd.Series(close_vec * volume_vec, index=date_index)
        .rolling(63)
        .mean()
        .iloc[-1]
    )
    assert signal_data_df[("AAA", ADV_63_FIELD_STR)].iloc[-1] == pytest.approx(
        expected_adv_float
    )


@pytest.mark.parametrize(
    ("raw_close_vec", "volume_vec", "expected_error_str"),
    [
        (None, [1_000.0, 1_000.0], "AAA.Unadjusted Close"),
        ([10.0, np.nan], [np.nan, 1_000.0], "no overlapping finite observations"),
        ([10.0, np.nan], [1_000.0, np.nan], "no usable ADV63 data"),
    ],
)
def test_relative_liquidity_fails_loudly_without_usable_raw_inputs(
    raw_close_vec,
    volume_vec,
    expected_error_str: str,
):
    date_index = pd.bdate_range("2024-01-02", periods=2)
    pricing_column_dict = {
        ("AAA", "High"): [11.0, 11.0],
        ("AAA", "Low"): [9.0, 9.0],
        ("AAA", "Close"): [10.0, 10.0],
        ("AAA", "Volume"): volume_vec,
    }
    if raw_close_vec is not None:
        pricing_column_dict[("AAA", "Unadjusted Close")] = raw_close_vec
    pricing_data_df = pd.DataFrame(pricing_column_dict, index=date_index)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = make_strategy(liquidity_mode_str=LIQUIDITY_RELATIVE_STR)

    with pytest.raises(RuntimeError, match=expected_error_str):
        strategy_obj.compute_signals(pricing_data_df)


def test_relative_liquidity_requires_adv63_in_execution_window():
    date_index = pd.bdate_range("2023-10-02", periods=64)
    raw_close_vec = np.full(len(date_index), 10.0)
    volume_vec = np.full(len(date_index), 1_000.0)
    raw_close_vec[-1] = np.nan
    volume_vec[-1] = np.nan
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "High"): 11.0,
            ("AAA", "Low"): 9.0,
            ("AAA", "Close"): 10.0,
            ("AAA", "Unadjusted Close"): raw_close_vec,
            ("AAA", "Volume"): volume_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = make_strategy(
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
        backtest_start_date_str=str(date_index[-1].date()),
    )

    with pytest.raises(RuntimeError, match="no usable ADV63 data"):
        strategy_obj.compute_signals(pricing_data_df)


def test_vote_and_relative_liquidity_entries_match_literal_rules():
    vote_strategy_obj = make_strategy(
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
    )
    liquid_strategy_obj = make_strategy(
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    )
    universe_df = pd.DataFrame(
        {"HIGH": [1], "LOW": [1]},
        index=[pd.Timestamp("2024-03-08")],
    )
    vote_strategy_obj.universe_df = universe_df
    liquid_strategy_obj.universe_df = universe_df
    row_value_dict = {}
    for symbol_str, turnover_float, adv_float in (
        ("HIGH", 30.0, 30_000_000.0),
        ("LOW", 20.0, 10_000_000.0),
    ):
        row_value_dict.update(
            eligible_field_dict(
                symbol_str,
                turnover_float=turnover_float,
                natr_float=3.0,
            )
        )
        row_value_dict.update(
            {
                (symbol_str, RETURN_2D_FIELD_STR): -0.02,
                (symbol_str, RETURN_5D_FIELD_STR): -0.04,
                (symbol_str, HPI_2D_FIELD_STR): 20.0,
                (symbol_str, HPI_5D_FIELD_STR): 20.0,
                (symbol_str, RAW_PRICE_FIELD_STR): 20.0,
                (symbol_str, ADV_63_FIELD_STR): adv_float,
            }
        )
    row_value_dict[("LOW", HPI_2D_FIELD_STR)] = 60.0
    row_value_dict[("LOW", HPI_5D_FIELD_STR)] = 60.0
    close_row_ser = make_close_row_ser(row_value_dict)
    member_symbol_set = {"HIGH", "LOW"}

    assert vote_strategy_obj.get_opportunity_list(
        close_row_ser,
        member_symbol_set,
    ) == ["HIGH"]
    assert liquid_strategy_obj.get_opportunity_list(
        close_row_ser,
        member_symbol_set,
    ) == ["HIGH"]


def test_missing_session_does_not_change_the_observation_clock():
    date_index = pd.bdate_range("2018-01-02", periods=1_265)
    step_vec = np.arange(len(date_index), dtype=float)
    close_vec = 100.0 + 0.03 * step_vec + 2.0 * np.sin(step_vec * 0.07)
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): close_vec - 0.2,
            ("AAA", "High"): close_vec + 1.0,
            ("AAA", "Low"): close_vec - 1.0,
            ("AAA", "Close"): close_vec,
            ("AAA", TURNOVER_FIELD_STR): 25_000_000.0 + step_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    missing_date_ts = date_index[500]
    no_padding_df = pricing_data_df.drop(index=missing_date_ts)
    padded_nan_df = pricing_data_df.copy()
    padded_nan_df.loc[missing_date_ts, :] = np.nan
    strategy_obj = make_strategy()

    no_padding_signal_df = strategy_obj.compute_signals(no_padding_df)
    padded_signal_df = strategy_obj.compute_signals(padded_nan_df)
    last_date_ts = date_index[-1]

    assert padded_signal_df.loc[
        last_date_ts,
        ("AAA", "return_3d_ser"),
    ] == pytest.approx(
        no_padding_signal_df.loc[last_date_ts, ("AAA", "return_3d_ser")]
    )
    assert padded_signal_df.loc[
        last_date_ts,
        ("AAA", "hpi_value_ser"),
    ] == pytest.approx(
        no_padding_signal_df.loc[last_date_ts, ("AAA", "hpi_value_ser")]
    )


def test_exact_loader_preserves_final_membership_and_uses_no_padding(monkeypatch):
    date_index = pd.bdate_range("2024-01-02", periods=8)
    membership_df = pd.DataFrame(
        {"Index Constituent": [1, 1, 1, 1, 1, 0, 0, 0]},
        index=date_index,
    )
    padding_obj_list: list[object] = []

    def price_timeseries_stub(symbol_str: str, **kwargs) -> pd.DataFrame:
        padding_obj_list.append(kwargs["padding_setting"])
        close_vec = np.arange(len(date_index), dtype=float) + 100.0
        price_df = pd.DataFrame(
            {
                "Open": close_vec,
                "High": close_vec + 1.0,
                "Low": close_vec - 1.0,
                "Close": close_vec,
                "Turnover": 1_000_000.0,
                "Dividend": 0.0,
            },
            index=date_index,
        )
        if symbol_str == "OLD":
            price_df = price_df.drop(index=date_index[3])
            price_df.loc[date_index[4], "Dividend"] = np.nan
        return price_df

    fake_norgatedata_obj = SimpleNamespace(
        watchlist_symbols=lambda _watchlist_str: ["OLD"],
        index_constituent_timeseries=lambda *_args, **_kwargs: membership_df,
        price_timeseries=price_timeseries_stub,
        StockPriceAdjustmentType=SimpleNamespace(
            TOTALRETURN="totalreturn",
            CAPITALSPECIAL="capitalspecial",
        ),
        PaddingType=SimpleNamespace(NONE="none"),
    )
    monkeypatch.setitem(sys.modules, "norgatedata", fake_norgatedata_obj)
    monkeypatch.setattr(
        "strategies.hpi.stateful_long.is_snapshot_mode_enabled_bool",
        lambda: False,
    )

    symbol_list, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str="S&P 500",
        benchmark_symbol_str="$SPX",
        start_date_str="2024-01-02",
        end_date_str="2024-01-11",
    )

    assert symbol_list == ["OLD"]
    assert universe_df.loc[date_index[4], "OLD"] == 1
    assert universe_df.loc[date_index[5], "OLD"] == 0
    assert padding_obj_list == ["none", "none"]
    assert ("OLD", "Close") in pricing_data_df.columns
    assert ("$SPX", "Close") in pricing_data_df.columns
    assert pd.isna(pricing_data_df.loc[date_index[3], ("OLD", "Open")])
    assert pricing_data_df.loc[date_index[3], ("OLD", "Close")] == pytest.approx(
        102.0
    )
    assert pricing_data_df.loc[
        date_index[3],
        ("OLD", "Dividend"),
    ] == pytest.approx(0.0)
    assert pd.isna(pricing_data_df.loc[date_index[4], ("OLD", "Dividend")])


def test_exact_loader_reads_dedicated_hpi_snapshot(tmp_path, monkeypatch):
    from data.norgate_snapshot_store import (
        CAPITALSPECIAL_ADJUSTMENT_STR,
        HPI_SP500_DATA_CONTRACT_DICT,
        HPI_SP500_PROFILE_STR,
        TOTALRETURN_ADJUSTMENT_STR,
        use_norgate_data_profile,
        write_snapshot_files,
    )

    date_idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    price_df = pd.DataFrame(
        [
            {
                "date": date_idx[0],
                "symbol_str": "OLD",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.0,
                "Close": 10.5,
                "Turnover": 1_000_000.0,
                "Dividend": 0.0,
            },
            *[
                {
                    "date": date_ts,
                    "symbol_str": "$SPX",
                    "adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Turnover": 0.0,
                    "Dividend": 0.0,
                }
                for date_ts in date_idx
            ],
        ]
    )
    universe_df = pd.DataFrame(
        {"OLD": [1, 0]},
        index=date_idx,
    )
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=HPI_SP500_PROFILE_STR,
        snapshot_date_str="2024-01-03",
        price_df=price_df,
        universe_df=universe_df,
        required_symbol_list=["OLD", "$SPX"],
        data_contract_dict=HPI_SP500_DATA_CONTRACT_DICT,
    )
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))

    with use_norgate_data_profile(HPI_SP500_PROFILE_STR):
        symbol_list, loaded_universe_df, pricing_data_df = (
            load_exact_hpi_inputs(
                indexname_str="S&P 500",
                benchmark_symbol_str="$SPX",
                start_date_str="2024-01-02",
                end_date_str="2024-01-03",
            )
        )

    assert symbol_list == ["OLD"]
    assert loaded_universe_df.loc[date_idx[1], "OLD"] == 0
    assert pd.isna(pricing_data_df.loc[date_idx[1], ("OLD", "Open")])
    assert pricing_data_df.loc[
        date_idx[1],
        ("OLD", "Close"),
    ] == pytest.approx(10.5)


def test_exact_loader_rejects_shared_snapshot_profile(monkeypatch):
    from data.norgate_snapshot_store import use_norgate_data_profile

    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    with use_norgate_data_profile("norgate_eod_sp500_pit"):
        with pytest.raises(RuntimeError, match="Strict HPI snapshot mode"):
            load_exact_hpi_inputs(
                indexname_str="S&P 500",
                benchmark_symbol_str="$SPX",
                start_date_str="2024-01-02",
                end_date_str="2024-01-03",
            )


@pytest.mark.parametrize(
    ("ranking_field_str", "expected_symbol_list"),
    [
        (TURNOVER_FIELD_STR, ["BBB", "AAA"]),
        (NATR_FIELD_STR, ["AAA", "BBB"]),
    ],
)
def test_opportunity_ranking_depends_only_on_approved_universe_rule(
    ranking_field_str: str,
    expected_symbol_list: list[str],
):
    strategy_obj = make_strategy(ranking_field_str=ranking_field_str)
    strategy_obj.previous_bar = pd.Timestamp("2024-03-08")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "OUT": [0]},
        index=[strategy_obj.previous_bar],
    )
    row_value_dict = {
        **eligible_field_dict("AAA", turnover_float=30.0, natr_float=5.0),
        **eligible_field_dict("BBB", turnover_float=50.0, natr_float=3.0),
        **eligible_field_dict("OUT", turnover_float=90.0, natr_float=9.0),
    }

    opportunity_symbol_list = strategy_obj.get_opportunity_list(
        make_close_row_ser(row_value_dict)
    )

    assert opportunity_symbol_list == expected_symbol_list


def test_removed_member_exit_funds_replacement_after_known_open_fill():
    strategy_obj = make_strategy(max_positions_int=1)
    strategy_obj.previous_bar = pd.Timestamp("2024-03-08")
    strategy_obj.current_bar = pd.Timestamp("2024-03-11")
    strategy_obj.universe_df = pd.DataFrame(
        {"OLD": [0], "NEW": [1]},
        index=[strategy_obj.previous_bar],
    )
    strategy_obj.add_transaction(
        7,
        pd.Timestamp("2024-03-07"),
        "OLD",
        10,
        100.0,
        1_000.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["OLD"] = 7
    row_value_dict = {
        **eligible_field_dict("NEW", turnover_float=50.0, natr_float=3.0),
        ("OLD", "Close"): 105.0,
        ("OLD", TURNOVER_FIELD_STR): 40.0,
        ("OLD", "return_3d_ser"): -0.02,
        ("OLD", "hpi_value_ser"): 20.0,
        ("OLD", "sma_200_price_ser"): 100.0,
        ("OLD", "ibs_value_ser"): np.nan,
        ("OLD", "rsi2_value_ser"): np.nan,
    }

    strategy_obj.iterate(
        pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)),
        make_close_row_ser(row_value_dict),
        pd.Series({"OLD": 104.0, "NEW": 106.0}),
    )

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 2
    exit_order_obj = next(order_obj for order_obj in order_list if order_obj.asset == "OLD")
    entry_order_obj = next(order_obj for order_obj in order_list if order_obj.asset == "NEW")
    assert isinstance(exit_order_obj, MarketOrder)
    assert exit_order_obj.target is True
    assert exit_order_obj.amount == 0.0
    assert exit_order_obj.trade_id == 7
    assert isinstance(entry_order_obj, MarketOrder)
    assert entry_order_obj.target is False
    assert entry_order_obj.amount == 100_000.0


def test_missing_open_for_current_member_defers_exit_until_next_tradable_open():
    strategy_obj = make_strategy()
    date_index = pd.bdate_range("2024-03-07", periods=4)
    strategy_obj.universe_df = pd.DataFrame(
        {"OLD": [1, 1, 1, 1]},
        index=date_index,
    )
    strategy_obj.add_transaction(
        7,
        date_index[0],
        "OLD",
        10,
        100.0,
        1_000.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["OLD"] = 7
    pricing_data_df = pd.DataFrame(
        {
            ("OLD", "Open"): [100.0, np.nan, np.nan, 98.0],
            ("OLD", "High"): [101.0, np.nan, np.nan, 99.0],
            ("OLD", "Low"): [99.0, np.nan, np.nan, 97.0],
            ("OLD", "Close"): [100.0, 100.0, 100.0, 98.5],
            ("OLD", "Dividend"): [0.0, 0.0, 0.0, 0.0],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    exit_signal_row_ser = make_close_row_ser(
        {
            ("OLD", "Close"): 100.0,
            ("OLD", TURNOVER_FIELD_STR): 40.0,
            ("OLD", "return_3d_ser"): -0.02,
            ("OLD", "hpi_value_ser"): 20.0,
            ("OLD", "sma_200_price_ser"): 90.0,
            ("OLD", "ibs_value_ser"): 0.95,
            ("OLD", "rsi2_value_ser"): np.nan,
        }
    )

    strategy_obj.previous_bar = date_index[0]
    strategy_obj.current_bar = date_index[1]
    strategy_obj.iterate(
        pricing_data_df.loc[:date_index[0]],
        exit_signal_row_ser,
        pd.Series({"OLD": np.nan}),
    )
    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.get_position("OLD") == pytest.approx(10.0)
    assert strategy_obj.get_orders() == []
    assert strategy_obj.pending_exit_symbol_set == {"OLD"}

    strategy_obj.previous_bar = date_index[1]
    strategy_obj.current_bar = date_index[2]
    strategy_obj.iterate(
        pricing_data_df.loc[:date_index[1]],
        exit_signal_row_ser,
        pd.Series({"OLD": np.nan}),
    )
    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.get_position("OLD") == pytest.approx(10.0)
    assert strategy_obj.get_orders() == []
    assert strategy_obj.pending_exit_symbol_set == {"OLD"}

    strategy_obj.previous_bar = date_index[2]
    strategy_obj.current_bar = date_index[3]
    strategy_obj.iterate(
        pricing_data_df.loc[:date_index[2]],
        exit_signal_row_ser,
        pd.Series({"OLD": 98.0}),
    )
    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.get_position("OLD") == pytest.approx(0.0)
    transaction_df = strategy_obj.get_transactions()
    assert transaction_df.iloc[-1]["bar"] == date_index[3]
    assert transaction_df.iloc[-1]["price"] == pytest.approx(98.0)


def test_missing_open_after_pit_removal_liquidates_at_last_close():
    strategy_obj = make_strategy()
    date_index = pd.bdate_range("2024-03-07", periods=3)
    strategy_obj.universe_df = pd.DataFrame(
        {"OLD": [1, 0, 0]},
        index=date_index,
    )
    strategy_obj.add_transaction(
        7,
        date_index[0],
        "OLD",
        10,
        100.0,
        1_000.0,
        1,
        0.0,
    )
    pricing_data_df = pd.DataFrame(
        {
            ("OLD", "Open"): [100.0, np.nan, np.nan],
            ("OLD", "High"): [101.0, np.nan, np.nan],
            ("OLD", "Low"): [99.0, np.nan, np.nan],
            ("OLD", "Close"): [100.0, 100.0, 100.0],
            ("OLD", "Dividend"): [0.0, np.nan, np.nan],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

    strategy_obj.previous_bar = date_index[0]
    strategy_obj.current_bar = date_index[1]
    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.get_position("OLD") == pytest.approx(0.0)
    liquidation_row_ser = strategy_obj.get_transactions().iloc[-1]
    assert liquidation_row_ser["bar"] == date_index[1]
    assert liquidation_row_ser["amount"] == pytest.approx(-10.0)
    assert liquidation_row_ser["price"] == pytest.approx(100.0)
    assert liquidation_row_ser["order_id"] == -1

    strategy_obj.previous_bar = date_index[1]
    strategy_obj.current_bar = date_index[2]
    strategy_obj.process_orders(pricing_data_df)


def test_missing_top_ranked_open_does_not_substitute_lower_ranked_entry():
    strategy_obj = make_strategy(max_positions_int=1)
    date_index = pd.bdate_range("2024-03-08", periods=2)
    strategy_obj.previous_bar = date_index[0]
    strategy_obj.current_bar = date_index[1]
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1]},
        index=[date_index[0]],
    )
    row_value_dict = {
        **eligible_field_dict("AAA", turnover_float=50.0, natr_float=3.0),
        **eligible_field_dict("BBB", turnover_float=40.0, natr_float=4.0),
    }
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [105.0, np.nan],
            ("AAA", "High"): [106.0, np.nan],
            ("AAA", "Low"): [104.0, np.nan],
            ("AAA", "Close"): [105.0, 105.0],
            ("BBB", "Open"): [105.0, 106.0],
            ("BBB", "High"): [106.0, 107.0],
            ("BBB", "Low"): [104.0, 105.0],
            ("BBB", "Close"): [105.0, 106.5],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

    strategy_obj.iterate(
        pricing_data_df.loc[:date_index[0]],
        make_close_row_ser(row_value_dict),
        pd.Series({"AAA": np.nan, "BBB": 106.0}),
    )

    assert [order_obj.asset for order_obj in strategy_obj.get_orders()] == ["AAA"]
    strategy_obj.process_orders(pricing_data_df)
    assert strategy_obj.get_position("AAA") == 0
    assert strategy_obj.get_position("BBB") == 0


@pytest.mark.parametrize(
    ("ibs_value_float", "rsi2_value_float"),
    [(0.91, 50.0), (0.50, 91.0)],
)
def test_overbought_exit_rules_are_stateful(
    ibs_value_float: float,
    rsi2_value_float: float,
):
    strategy_obj = make_strategy()
    strategy_obj.previous_bar = pd.Timestamp("2024-03-08")
    strategy_obj.current_bar = pd.Timestamp("2024-03-11")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1]},
        index=[strategy_obj.previous_bar],
    )
    strategy_obj.add_transaction(
        9,
        pd.Timestamp("2024-03-07"),
        "AAA",
        10,
        100.0,
        1_000.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["AAA"] = 9
    row_value_dict = {
        ("AAA", "Close"): 105.0,
        ("AAA", TURNOVER_FIELD_STR): 40.0,
        ("AAA", "return_3d_ser"): 0.02,
        ("AAA", "hpi_value_ser"): 60.0,
        ("AAA", "sma_200_price_ser"): 100.0,
        ("AAA", "ibs_value_ser"): ibs_value_float,
        ("AAA", "rsi2_value_ser"): rsi2_value_float,
    }

    strategy_obj.iterate(
        pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)),
        make_close_row_ser(row_value_dict),
        pd.Series({"AAA": 106.0}),
    )

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 1
    assert order_list[0].asset == "AAA"
    assert order_list[0].target is True
    assert order_list[0].amount == 0.0


def test_position_remains_open_without_an_exit_signal():
    strategy_obj = make_strategy()
    strategy_obj.previous_bar = pd.Timestamp("2024-03-08")
    strategy_obj.current_bar = pd.Timestamp("2024-03-11")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1]},
        index=[strategy_obj.previous_bar],
    )
    strategy_obj.add_transaction(
        3,
        pd.Timestamp("2024-03-07"),
        "AAA",
        10,
        100.0,
        1_000.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["AAA"] = 3
    row_value_dict = {
        ("AAA", "Close"): 105.0,
        ("AAA", TURNOVER_FIELD_STR): 40.0,
        ("AAA", "return_3d_ser"): 0.01,
        ("AAA", "hpi_value_ser"): 60.0,
        ("AAA", "sma_200_price_ser"): 100.0,
        ("AAA", "ibs_value_ser"): 0.50,
        ("AAA", "rsi2_value_ser"): 50.0,
    }

    strategy_obj.iterate(
        pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)),
        make_close_row_ser(row_value_dict),
        pd.Series(dtype=float),
    )

    assert strategy_obj.get_orders() == []


def test_bench_discovers_hpi_variants_with_expected_wired_scope():
    expected_module_set = {
        "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit",
        "strategies.hpi.strategy_mr_hpi_nasdaq100_ibs_rsi_exit",
        "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote",
        (
            "strategies.hpi."
            "strategy_mr_hpi_sp500_2_3_5_vote_relative_liquidity"
        ),
    }
    hpi_entry_list = [
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.module_import_str in expected_module_set
    ]

    assert {entry_obj.module_import_str for entry_obj in hpi_entry_list} == expected_module_set
    assert all(entry_obj.category_label_str == "HPI mean-reversion" for entry_obj in hpi_entry_list)
    assert all(entry_obj.has_run_variant_bool for entry_obj in hpi_entry_list)
    wired_module_set = {
        entry_obj.module_import_str
        for entry_obj in hpi_entry_list
        if entry_obj.is_wired_bool
    }
    assert wired_module_set == {
        "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote",
        "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit",
    }


@pytest.mark.parametrize(
    ("module_import_str", "expected_kwarg_dict"),
    [
        (
            "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote",
            {
                "entry_mode_str": ENTRY_HORIZON_VOTE_STR,
                "benchmark_symbol_str": "$SPXTR",
            },
        ),
        (
            "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit",
            {
                "benchmark_symbol_str": "$SPXTR",
            },
        ),
        (
            (
                "strategies.hpi."
                "strategy_mr_hpi_sp500_2_3_5_vote_relative_liquidity"
            ),
            {
                "entry_mode_str": ENTRY_HORIZON_VOTE_STR,
                "liquidity_mode_str": LIQUIDITY_RELATIVE_STR,
                "benchmark_symbol_str": "$SPX",
            },
        ),
    ],
)
def test_new_bench_wrappers_forward_exact_variant_mode(
    monkeypatch,
    module_import_str: str,
    expected_kwarg_dict: dict[str, str],
):
    strategy_module = importlib.import_module(module_import_str)
    captured_kwarg_dict = {}

    def fake_run_hpi_variant(**kwarg_dict):
        captured_kwarg_dict.update(kwarg_dict)
        return "strategy"

    monkeypatch.setattr(
        strategy_module,
        "run_hpi_variant",
        fake_run_hpi_variant,
    )

    assert strategy_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
    ) == "strategy"
    assert captured_kwarg_dict["indexname_str"] == "S&P 500"
    assert captured_kwarg_dict["ranking_field_str"] == TURNOVER_FIELD_STR
    for kwarg_str, expected_value_str in expected_kwarg_dict.items():
        assert captured_kwarg_dict[kwarg_str] == expected_value_str


def test_run_hpi_variant_records_adjustment_provenance(monkeypatch):
    date_index = pd.bdate_range("2024-01-02", periods=2)
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Close"): [10.0, 11.0],
            ("$SPX", "Close"): [4_700.0, 4_710.0],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    universe_df = pd.DataFrame({"AAA": [1, 1]}, index=date_index)
    captured_adjustment_dict = {}

    monkeypatch.setattr(
        "strategies.hpi.stateful_long.load_exact_hpi_inputs",
        lambda **_kwarg_dict: (["AAA"], universe_df, pricing_data_df),
    )

    def fake_run_daily(
        _strategy_obj,
        received_pricing_data_df,
        _calendar_idx,
        **_kwarg_dict,
    ):
        captured_adjustment_dict.update(
            received_pricing_data_df.attrs[
                "norgate_adjustment_by_symbol_dict"
            ]
        )

    monkeypatch.setattr(
        "strategies.hpi.stateful_long.run_daily",
        fake_run_daily,
    )

    run_hpi_variant(
        strategy_name_str="HPIAdjustmentProvenanceTest",
        indexname_str="S&P 500",
        benchmark_symbol_str="$SPX",
        ranking_field_str=TURNOVER_FIELD_STR,
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-02",
    )

    assert captured_adjustment_dict == {
        "AAA": "CAPITALSPECIAL",
        "$SPX": "TOTALRETURN",
    }
