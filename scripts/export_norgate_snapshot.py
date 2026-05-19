from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    write_snapshot_files,
)


@dataclass(frozen=True)
class NorgateExportProfileSpec:
    indexname_str: str | None = None
    capital_symbol_tuple: tuple[str, ...] = ()
    total_return_symbol_tuple: tuple[str, ...] = ()
    helper_symbol_tuple: tuple[str, ...] = ()


PROFILE_EXPORT_SPEC_DICT: dict[str, NorgateExportProfileSpec] = {
    "norgate_eod_sp500_pit": NorgateExportProfileSpec(
        indexname_str="S&P 500",
        total_return_symbol_tuple=("$SPX",),
    ),
    "norgate_eod_etf_plus_vix_helper": NorgateExportProfileSpec(
        capital_symbol_tuple=("GLD", "UUP", "TLT", "DBC", "BTAL", "SPY", "QQQ", "TQQQ"),
        total_return_symbol_tuple=("GLD", "UUP", "TLT", "DBC", "BTAL", "$SPX"),
        helper_symbol_tuple=("$VIX",),
    ),
    "norgate_eod_ndx_pit": NorgateExportProfileSpec(
        indexname_str="Nasdaq 100",
        capital_symbol_tuple=("SPY",),
    ),
    "norgate_eod_ndx_pit_plus_vxn_helper": NorgateExportProfileSpec(
        indexname_str="Nasdaq 100",
        capital_symbol_tuple=("SPY",),
        helper_symbol_tuple=("$VXN",),
    ),
}

SUPPORTED_EOD_PROFILE_TUPLE: tuple[str, ...] = tuple(PROFILE_EXPORT_SPEC_DICT.keys())


def _load_direct_norgate_module():
    try:
        import norgatedata as norgatedata_module
    except ModuleNotFoundError as exc:
        raise RuntimeError("Snapshot export must run on the Windows Norgate node.") from exc
    return norgatedata_module


def validate_export_profile(profile_str: str) -> None:
    if profile_str not in PROFILE_EXPORT_SPEC_DICT:
        raise ValueError(
            f"Unsupported EOD Norgate export profile '{profile_str}'. "
            f"Expected one of {SUPPORTED_EOD_PROFILE_TUPLE}."
        )


def _latest_norgate_session_date_str() -> str:
    norgatedata_module = _load_direct_norgate_module()
    heartbeat_df = norgatedata_module.price_timeseries("$SPX", timeseriesformat="pandas-dataframe")
    if heartbeat_df is None or len(heartbeat_df.index) == 0:
        raise RuntimeError("Norgate returned no $SPX heartbeat data.")
    return pd.Timestamp(heartbeat_df.index[-1]).date().isoformat()


def _load_index_constituent_matrix_df(indexname_str: str) -> tuple[list[str], pd.DataFrame]:
    norgatedata_module = _load_direct_norgate_module()
    symbol_list = norgatedata_module.watchlist_symbols(f"{indexname_str} Current & Past")
    calendar_index = norgatedata_module.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index
    last_trading_day_ts = calendar_index[-1]
    universe_frame_list: list[pd.DataFrame] = []

    for symbol_str in tqdm(symbol_list, desc=f"exporting {indexname_str} universe"):
        constituent_df = norgatedata_module.index_constituent_timeseries(
            symbol_str,
            indexname_str,
            timeseriesformat="pandas-dataframe",
        )
        if constituent_df["Index Constituent"].sum() <= 0:
            continue
        constituent_df = constituent_df.rename(columns={"Index Constituent": symbol_str})
        constituent_df = constituent_df.loc[constituent_df[symbol_str] == 1]
        if last_trading_day_ts != constituent_df.index[-1]:
            constituent_df = constituent_df.iloc[:-5]
        universe_frame_list.append(constituent_df)

    if len(universe_frame_list) == 0:
        raise RuntimeError(f"No PIT universe rows were exported for {indexname_str}.")
    universe_df = pd.concat(universe_frame_list, axis=1).fillna(0).astype(int).sort_index()
    return [str(symbol_str) for symbol_str in universe_df.columns.tolist()], universe_df


def _adjustment_type_obj(adjustment_str: str):
    norgatedata_module = _load_direct_norgate_module()
    if adjustment_str == TOTALRETURN_ADJUSTMENT_STR:
        return norgatedata_module.StockPriceAdjustmentType.TOTALRETURN
    return norgatedata_module.StockPriceAdjustmentType.CAPITALSPECIAL


def _load_price_frame_df(
    *,
    symbol_str: str,
    adjustment_str: str,
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    norgatedata_module = _load_direct_norgate_module()
    raw_price_df = norgatedata_module.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=_adjustment_type_obj(adjustment_str),
        padding_setting=norgatedata_module.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if raw_price_df is None or len(raw_price_df.index) == 0:
        raise RuntimeError(f"Norgate returned no price data for {symbol_str} ({adjustment_str}).")

    price_df = raw_price_df.reset_index()
    price_df = price_df.rename(columns={price_df.columns[0]: "date"})
    price_df.insert(1, "symbol_str", symbol_str)
    price_df.insert(2, "adjustment_str", adjustment_str)
    return price_df


def _build_price_snapshot_df(
    *,
    capital_symbol_list: Iterable[str],
    total_return_symbol_list: Iterable[str],
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    price_frame_list: list[pd.DataFrame] = []
    for symbol_str in tqdm(list(dict.fromkeys(capital_symbol_list)), desc="exporting CAPITALSPECIAL prices"):
        price_frame_list.append(
            _load_price_frame_df(
                symbol_str=symbol_str,
                adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
            )
        )
    for symbol_str in tqdm(list(dict.fromkeys(total_return_symbol_list)), desc="exporting TOTALRETURN prices"):
        price_frame_list.append(
            _load_price_frame_df(
                symbol_str=symbol_str,
                adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
            )
        )
    if len(price_frame_list) == 0:
        raise RuntimeError("No price rows were exported.")
    return pd.concat(price_frame_list, axis=0, ignore_index=True)


def _build_adjustment_mode_map_dict(
    *,
    capital_symbol_list: Iterable[str],
    total_return_symbol_list: Iterable[str],
) -> dict[str, str | list[str]]:
    adjustment_mode_map_dict: dict[str, list[str]] = {}
    for symbol_str in capital_symbol_list:
        adjustment_mode_map_dict.setdefault(str(symbol_str), []).append(CAPITALSPECIAL_ADJUSTMENT_STR)
    for symbol_str in total_return_symbol_list:
        adjustment_mode_map_dict.setdefault(str(symbol_str), []).append(TOTALRETURN_ADJUSTMENT_STR)

    collapsed_adjustment_mode_map_dict: dict[str, str | list[str]] = {}
    for symbol_str, adjustment_list in adjustment_mode_map_dict.items():
        unique_adjustment_list = list(dict.fromkeys(adjustment_list))
        if len(unique_adjustment_list) == 1:
            collapsed_adjustment_mode_map_dict[symbol_str] = unique_adjustment_list[0]
        else:
            collapsed_adjustment_mode_map_dict[symbol_str] = unique_adjustment_list
    return collapsed_adjustment_mode_map_dict


def _export_profile_to_root_path(
    *,
    snapshot_root_str: str,
    profile_str: str,
    snapshot_date_str: str,
    start_date_str: str,
    end_date_str: str,
    overwrite_bool: bool,
) -> Path:
    validate_export_profile(profile_str)
    profile_spec_obj = PROFILE_EXPORT_SPEC_DICT[profile_str]
    universe_df: pd.DataFrame | None = None
    pit_symbol_list: list[str] = []
    if profile_spec_obj.indexname_str is not None:
        pit_symbol_list, universe_df = _load_index_constituent_matrix_df(profile_spec_obj.indexname_str)

    symbol_list = list(dict.fromkeys(pit_symbol_list + list(profile_spec_obj.capital_symbol_tuple)))
    helper_symbol_list = list(profile_spec_obj.helper_symbol_tuple)
    benchmark_symbol_list = list(profile_spec_obj.total_return_symbol_tuple)
    price_df = _build_price_snapshot_df(
        capital_symbol_list=symbol_list + helper_symbol_list,
        total_return_symbol_list=benchmark_symbol_list,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )

    required_symbol_list = list(dict.fromkeys(symbol_list + benchmark_symbol_list))
    adjustment_mode_map_dict = _build_adjustment_mode_map_dict(
        capital_symbol_list=symbol_list + helper_symbol_list,
        total_return_symbol_list=benchmark_symbol_list,
    )
    return write_snapshot_files(
        snapshot_root_str=snapshot_root_str,
        profile_str=profile_str,
        snapshot_date_str=snapshot_date_str,
        price_df=price_df,
        universe_df=universe_df,
        required_symbol_list=required_symbol_list,
        required_helper_symbol_list=helper_symbol_list,
        adjustment_mode_map_dict=adjustment_mode_map_dict,
        overwrite_bool=overwrite_bool,
    )


def export_profile_snapshot(
    *,
    snapshot_root_str: str,
    profile_str: str,
    snapshot_date_str: str | None = None,
    start_date_str: str = "1990-01-01",
    end_date_str: str | None = None,
    overwrite_bool: bool = False,
) -> Path:
    validate_export_profile(profile_str)
    resolved_snapshot_date_str = snapshot_date_str or _latest_norgate_session_date_str()
    resolved_end_date_str = str(end_date_str or resolved_snapshot_date_str)
    snapshot_root_path_obj = Path(snapshot_root_str).expanduser()
    final_snapshot_dir_path_obj = snapshot_root_path_obj / profile_str / resolved_snapshot_date_str
    if final_snapshot_dir_path_obj.exists() and not overwrite_bool:
        manifest_path_obj = final_snapshot_dir_path_obj / "manifest.json"
        if manifest_path_obj.exists():
            return final_snapshot_dir_path_obj
        raise FileExistsError(
            f"Snapshot directory exists without a manifest; remove or overwrite it: {final_snapshot_dir_path_obj}"
        )

    staging_root_path_obj = (
        snapshot_root_path_obj
        / f".staging-{profile_str}-{resolved_snapshot_date_str}-{os.getpid()}"
    )
    if staging_root_path_obj.exists():
        shutil.rmtree(staging_root_path_obj)

    try:
        staged_snapshot_dir_path_obj = _export_profile_to_root_path(
            snapshot_root_str=str(staging_root_path_obj),
            profile_str=profile_str,
            snapshot_date_str=resolved_snapshot_date_str,
            start_date_str=start_date_str,
            end_date_str=resolved_end_date_str,
            overwrite_bool=False,
        )

        final_profile_dir_path_obj = final_snapshot_dir_path_obj.parent
        final_profile_dir_path_obj.mkdir(parents=True, exist_ok=True)
        if final_snapshot_dir_path_obj.exists():
            shutil.rmtree(final_snapshot_dir_path_obj)
        staged_snapshot_dir_path_obj.rename(final_snapshot_dir_path_obj)
        return final_snapshot_dir_path_obj
    finally:
        if staging_root_path_obj.exists():
            shutil.rmtree(staging_root_path_obj)


def main() -> int:
    parser_obj = argparse.ArgumentParser(description="Export a validated Norgate snapshot on the Windows node.")
    parser_obj.add_argument("--snapshot-root", required=True, help="Output norgate_snapshots root.")
    parser_obj.add_argument("--profile", required=True, help="Snapshot profile name.")
    parser_obj.add_argument("--snapshot-date", default=None, help="Market-session date, YYYY-MM-DD.")
    parser_obj.add_argument("--start-date", default="1990-01-01", help="First date to export.")
    parser_obj.add_argument("--end-date", default=None, help="Last date to export. Defaults to snapshot date.")
    parser_obj.add_argument("--overwrite", action="store_true", help="Replace an existing snapshot directory.")
    args_obj = parser_obj.parse_args()

    snapshot_dir_path_obj = export_profile_snapshot(
        snapshot_root_str=str(args_obj.snapshot_root),
        profile_str=str(args_obj.profile),
        snapshot_date_str=None if args_obj.snapshot_date is None else str(args_obj.snapshot_date),
        start_date_str=str(args_obj.start_date),
        end_date_str=None if args_obj.end_date is None else str(args_obj.end_date),
        overwrite_bool=bool(args_obj.overwrite),
    )
    print(snapshot_dir_path_obj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
