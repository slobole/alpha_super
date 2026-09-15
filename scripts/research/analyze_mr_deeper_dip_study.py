"""Saved-artifact analysis for the frozen MR deeper-dip experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from scripts.research.run_mr_deeper_dip_study import (
    STUDY_PATH, DEPTH_TUPLE, STRATEGY_TUPLE, BASE_COST_FLOAT,
    load_inputs, limit_fill_price, policy_name, write_json,
)

TABLE_PATH = STUDY_PATH / "tables"
CHART_PATH = STUDY_PATH / "charts"
DISPLAY_NAME_DICT={"dv2":"DV2","hpi235":"HPI vote (2/3/5)","sector":"Sector ETFs (VOX/IYR)"}
DISPLAY_POLICY_DICT={"moo":"Open (MOO)","limit_0.5pct":"Limit -0.5%","limit_1pct":"Limit -1%","limit_2pct":"Limit -2%"}
PERIOD_TUPLE = (
    ("2010-2014", "2010-01-05", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-09-14"),
)


def nav_returns(nav_ser: pd.Series, initial_float=100000.) -> pd.Series:
    # *** CRITICAL*** Backward-only accounting ratio E_t/E_(t-1)-1.
    # First observation uses declared initial cash, not a future/filled return.
    prior_ser = nav_ser.shift(1)
    prior_ser.iloc[0] = initial_float
    return nav_ser/prior_ser-1.


def performance(return_ser: pd.Series) -> dict:
    if return_ser.isna().any() or len(return_ser) < 2:
        raise AssertionError("Returns must be observed and complete.")
    wealth_vec = np.r_[1., (1.+return_ser).cumprod().to_numpy()]
    vol_float = float(return_ser.std(ddof=1)*np.sqrt(252))
    return {
        "cagr": float(wealth_vec[-1]**(252./len(return_ser))-1.),
        "volatility": vol_float,
        "sharpe": float(return_ser.mean()*252./vol_float) if vol_float else 0.,
        "max_drawdown": float(np.min(wealth_vec/np.maximum.accumulate(wealth_vec)-1.)),
        "total_return": float(wealth_vec[-1]-1.),
        "worst_day": float(return_ser.min()),
    }


def read_cell(cell_path: Path):
    if not (cell_path/"complete.json").exists():
        raise FileNotFoundError("Incomplete cell: "+str(cell_path))
    daily_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=cell_path/"daily.csv", index_col="date", parse_dates=True)
    entry_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=cell_path/"entries.csv")
    entry_df["date"] = pd.to_datetime(entry_df["date"])
    entry_df["decision_date"] = pd.to_datetime(entry_df["decision_date"])
    transaction_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=cell_path/"transactions.csv", parse_dates=["bar"])
    dividend_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=cell_path/"dividends.csv", parse_dates=["entitlement_date", "ex_date"])
    friction_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=cell_path/"friction.csv", parse_dates=["date"])
    return daily_df, entry_df, transaction_df, dividend_df, friction_df


def reconcile_cash(daily_df, transaction_df, dividend_df, friction_df):
    movement_ser = -transaction_df.groupby("bar")[["total_value", "commission"]].sum().sum(axis=1)
    dividend_ser = dividend_df.groupby("ex_date")["net_dividend_cash_float"].sum()
    friction_ser = friction_df.groupby("date")["research_friction"].sum()
    # *** CRITICAL*** Dates join realized cash events AFTER execution, not inputs.
    # Zero means no recorded cash event on a fully observed session.
    movement_ser = (movement_ser.reindex(daily_df.index, fill_value=0.)
                    +dividend_ser.reindex(daily_df.index, fill_value=0.)
                    -friction_ser.reindex(daily_df.index, fill_value=0.))
    expected_ser = 100000.+movement_ser.cumsum()
    error_float = float((expected_ser-daily_df["cash"]).abs().max())
    if error_float > 0.00001:
        raise AssertionError(f"Daily cash ledger mismatch: {error_float}")
    if not np.allclose(daily_df["cash"]+daily_df["invested"], daily_df["nav"], rtol=0, atol=1e-7):
        raise AssertionError("NAV != cash+holdings.")
    return error_float


def reconcile_daily_marks(daily_df, transaction_df, pricing_df):
    if not np.allclose(transaction_df["total_value"],
                       transaction_df["amount"]*transaction_df["price"],rtol=0,atol=1e-7):
        raise AssertionError("Transaction notional != shares*fill price.")
    quantity_delta_df=transaction_df.pivot_table(index="bar",columns="asset",
                                                values="amount",aggfunc="sum",fill_value=0.)
    # *** CRITICAL*** Post-execution accounting: missing transaction dates mean
    # zero changes, not interpolated prices or future signal information.
    quantity_df=quantity_delta_df.reindex(daily_df.index,fill_value=0.).cumsum()
    close_df=pricing_df.xs("Close",axis=1,level=1).reindex(
        index=daily_df.index,columns=quantity_df.columns)
    held_mask=~np.isclose(quantity_df,0.)
    if (held_mask&~np.isfinite(close_df)).any().any():
        raise AssertionError("Held asset has no final daily valuation price.")
    # Zero prices only where quantity is zero; held missing marks already fail.
    marked_ser=(quantity_df.where(held_mask,0.)*close_df.where(held_mask,0.)).sum(axis=1)
    error_float=float((marked_ser-daily_df["invested"]).abs().max())
    if error_float>1e-5:
        raise AssertionError(f"Independent daily close marks mismatch: {error_float}")
    return error_float


def verify_dividend_entitlements(calendar_idx, pricing_df, transaction_df, dividend_df):
    expected_map={}
    held_dict={}
    grouped_dict=dict(tuple(transaction_df.groupby("bar")))
    for bar_ts in calendar_idx:
        prior_ts=pricing_df.index[pricing_df.index.get_loc(bar_ts)-1]
        # *** CRITICAL*** Dividend belongs to shares held at prior close.
        # Reconstruct entitlement BEFORE applying current-session executions.
        for asset_str,shares_float in held_dict.items():
            if np.isclose(shares_float,0.):
                continue
            distribution_float=float(pricing_df.loc[prior_ts,(asset_str,"Dividend")])
            if not np.isfinite(distribution_float):
                raise AssertionError("Unknown dividend on a held entitlement session.")
            if not np.isclose(distribution_float,0.):
                gross_float=shares_float*distribution_float
                tax_float=max(gross_float,0.)*.25
                expected_map[(str(bar_ts.date()),asset_str)]=(
                    str(prior_ts.date()),shares_float,distribution_float,
                    gross_float,tax_float,gross_float-tax_float)
        if bar_ts in grouped_dict:
            for transaction_row in grouped_dict[bar_ts].itertuples(index=False):
                held_dict[transaction_row.asset]=held_dict.get(transaction_row.asset,0.)+transaction_row.amount
    actual_list=[(str(row.ex_date.date()),row.asset_str,
        (str(row.entitlement_date.date()),row.position_share_float,row.dividend_per_share_float,
         row.gross_dividend_cash_float,row.withholding_cash_float,row.net_dividend_cash_float))
        for row in dividend_df.itertuples(index=False)]
    actual_key_list=[(day_str,asset_str) for day_str,asset_str,_ in actual_list]
    if len(set(actual_key_list))!=len(actual_key_list) or set(actual_key_list)!=set(expected_map):
        raise AssertionError("Dividend event keys are duplicated or missing.")
    for day_str,asset_str,value_tuple in actual_list:
        expected_tuple=expected_map[(day_str,asset_str)]
        if value_tuple[0]!=expected_tuple[0] or not np.allclose(
                value_tuple[1:],expected_tuple[1:],rtol=0,atol=1e-8):
            raise AssertionError("Dividend entitlement fields disagree with independent holdings.")
    return len(actual_list)



def make_trade_table(transaction_df, dividend_df, friction_df, pricing_df):
    row_list = []
    last_ts = pd.Timestamp(pricing_df.index[-1])
    for trade_id_int, trade_df in transaction_df.groupby("trade_id", sort=False):
        entry_df = trade_df.loc[trade_df["amount"]>0.]
        if len(entry_df) != 1:
            raise AssertionError("Frozen strategies should have one entry per trade ID.")
        entry_row = entry_df.iloc[0]
        asset_str = str(entry_row["asset"])
        entry_ts = pd.Timestamp(entry_row["bar"])
        exit_df = trade_df.loc[trade_df["amount"]<0.]
        residual_float = float(trade_df["amount"].sum())
        if residual_float < -1e-7:
            raise AssertionError("Unexpected short exposure.")
        closed_bool = abs(residual_float) < 1e-7
        exit_ts = pd.Timestamp(exit_df["bar"].max()) if closed_bool else last_ts
        last_price_float = float(pricing_df.loc[last_ts, (asset_str, "Close")])
        terminal_value_float = residual_float*last_price_float if not closed_bool else 0.
        sale_value_float = -float(exit_df["total_value"].sum())+terminal_value_float
        quantity_float = float(entry_row["amount"])
        exit_equivalent_float = sale_value_float/quantity_float
        dividend_mask = ((dividend_df["asset_str"]==asset_str)
                         &(dividend_df["ex_date"]>entry_ts)&(dividend_df["ex_date"]<=exit_ts))
        dividend_float = float(dividend_df.loc[dividend_mask, "net_dividend_cash_float"].sum())
        friction_float = float(friction_df.loc[friction_df["trade_id"]==trade_id_int, "research_friction"].sum())
        native_fee_float = float(trade_df["commission"].sum())
        pnl_float = sale_value_float-float(entry_row["total_value"])+dividend_float-native_fee_float-friction_float
        row_list.append({
            "trade_id": trade_id_int, "asset": asset_str,
            "entry_date": entry_ts, "exit_date": exit_ts, "closed": closed_bool,
            "shares": quantity_float, "entry_price": float(entry_row["price"]),
            "exit_equivalent_price": exit_equivalent_float,
            "native_commission": native_fee_float, "research_friction": friction_float,
            "dividends": dividend_float, "net_pnl": pnl_float,
            "net_return": pnl_float/float(entry_row["total_value"]),
            "terminal_mark_value": terminal_value_float,
            "duration_sessions": int(pricing_df.index.searchsorted(exit_ts)
                                     -pricing_df.index.searchsorted(entry_ts)+1),
        })
    return pd.DataFrame(row_list)


def paired_opportunities(strategy_str, pricing_df, baseline_entry_df, baseline_trade_df):
    row_list = []
    matched_df = baseline_trade_df.merge(
        baseline_entry_df[["trade_id","open","low","sizing_price","decision_date"]],
        on="trade_id", validate="one_to_one")
    for trade_row in matched_df.itertuples(index=False):
        quantity_float = float(trade_row.shares)
        base_price_pnl_float = quantity_float*(trade_row.exit_equivalent_price-trade_row.open)
        base_net_pnl_float = float(trade_row.net_pnl)
        for depth_float in DEPTH_TUPLE[1:]:
            fill_float = limit_fill_price(trade_row.open, trade_row.low, depth_float)
            limit_pnl_float = 0.
            improvement_float = 0.
            if fill_float is not None:
                friction_float = quantity_float*fill_float*BASE_COST_FLOAT
                limit_pnl_float = (quantity_float*(trade_row.exit_equivalent_price-fill_float)
                                   +trade_row.dividends-trade_row.native_commission-friction_float)
                improvement_float = quantity_float*(trade_row.open-fill_float)
            row_list.append({
                "strategy": strategy_str, "policy": policy_name(depth_float),
                "trade_id": trade_row.trade_id, "asset": trade_row.asset,
                "date": trade_row.entry_date, "exit_date": trade_row.exit_date,
                "closed": trade_row.closed, "filled": fill_float is not None,
                "shares": quantity_float, "limit_price": fill_float,
                "baseline_net_pnl": base_net_pnl_float, "limit_net_pnl": limit_pnl_float,
                "delta_net_pnl": limit_pnl_float-base_net_pnl_float,
                "gross_price_improvement_on_fill": improvement_float,
                "missed_baseline_net_pnl": base_net_pnl_float if fill_float is None else 0.,
                "baseline_open_price_pnl": base_price_pnl_float,
                "baseline_return": base_net_pnl_float/(quantity_float*trade_row.open),
                "limit_return_or_zero": limit_pnl_float/(quantity_float*trade_row.open),
                "limit_return_on_fill_notional": limit_pnl_float/(quantity_float*fill_float) if fill_float else np.nan,
                "overnight_price_return": trade_row.open/trade_row.sizing_price-1.,
                "open_to_exit_price_return": trade_row.exit_equivalent_price/trade_row.open-1.,
                "close_to_exit_price_return": trade_row.exit_equivalent_price/trade_row.sizing_price-1.,
            })
    return pd.DataFrame(row_list)


def verify_execution_contract(strategy_str, layer_str, depth_float, entry_df,
                              transaction_df, friction_df, pricing_df):
    if entry_df["order_id"].duplicated().any():
        raise AssertionError("An entry order was recorded twice.")
    if transaction_df["order_id"].ge(0).sum() != transaction_df.loc[
            transaction_df["order_id"].ge(0),"order_id"].nunique():
        raise AssertionError("Duplicate ordinary execution.")
    pd.testing.assert_frame_equal(
        transaction_df[["bar","asset","trade_id","order_id","amount","price","commission"]].reset_index(drop=True),
        friction_df[["date","asset","trade_id","order_id","shares","fill_price","native_commission"]]
        .set_axis(["bar","asset","trade_id","order_id","amount","price","commission"],axis=1).reset_index(drop=True),
        check_dtype=False, rtol=0, atol=1e-9)
    filled_df = entry_df.loc[entry_df["status"].eq("filled")]
    buy_df = transaction_df.loc[transaction_df["amount"]>0.]
    joined_df = filled_df.merge(buy_df, on="order_id", suffixes=("_intent","_execution"),
                               validate="one_to_one")
    if len(joined_df)!=len(filled_df) or len(joined_df)!=len(buy_df):
        raise AssertionError("Entry intents and buy executions disagree.")
    if not np.allclose(joined_df["shares"],joined_df["amount"],rtol=0,atol=1e-9):
        raise AssertionError("Quantity changed after decision.")
    missed_id_set = set(entry_df.loc[~entry_df["status"].eq("filled"),"order_id"])
    if missed_id_set.intersection(transaction_df["order_id"]):
        raise AssertionError("An unfilled order generated a transaction.")
    expected_share_ser = (entry_df["prior_nav"]*(1.5/11.)/entry_df["sizing_price"]
                          if strategy_str=="sector" else
                          np.floor((entry_df["prior_nav"]/10.)/entry_df["sizing_price"]))
    if not np.allclose(entry_df["shares"],expected_share_ser,rtol=1e-12,atol=1e-8):
        raise AssertionError("Original prior-close sizing contract changed.")
    for order_row in entry_df.itertuples(index=False):
        expected_float = (order_row.open*(1.+BASE_COST_FLOAT) if depth_float is None
                          and np.isfinite(order_row.open) and order_row.open>0. else None)
        if depth_float is not None:
            expected_float = limit_fill_price(order_row.open,order_row.low,depth_float,
                                             .0005 if layer_str=="stress" else 0.)
        if (expected_float is not None) != (order_row.status=="filled"):
            raise AssertionError("Fill eligibility differs from frozen contract.")
        if expected_float is not None and not np.isclose(expected_float,order_row.fill_price,
                                                         rtol=0,atol=1e-9):
            raise AssertionError("Fill price differs from frozen contract.")
    expected_friction_ser = transaction_df["total_value"].abs()*(.001 if layer_str=="stress" else 0.)
    if depth_float is not None:
        expected_friction_ser += transaction_df["total_value"].clip(lower=0.)*BASE_COST_FLOAT
    if not np.allclose(expected_friction_ser,friction_df["research_friction"],rtol=0,atol=1e-8):
        raise AssertionError("Friction was omitted or charged twice.")
    expected_fee_ser = (transaction_df["amount"].abs()*.005).clip(lower=1.)
    if not np.allclose(expected_fee_ser,transaction_df["commission"],rtol=0,atol=1e-9):
        raise AssertionError("Native commission contract changed.")
    for trade_row in transaction_df.loc[
            (transaction_df["amount"]<0.)&(transaction_df["order_id"]>=0)].itertuples(index=False):
        expected_float=float(pricing_df.loc[trade_row.bar,(trade_row.asset,"Open")])*(1.-BASE_COST_FLOAT)
        if not np.isclose(expected_float,trade_row.price,rtol=0,atol=1e-9):
            raise AssertionError("Original exit execution price changed.")


def verify_complete_fingerprints(strategy_str, cell_path):
    import json
    from scripts.research.run_mr_deeper_dip_study import sha256_file
    complete_dict=json.loads((cell_path/"complete.json").read_text())
    path_dict={
        "code_sha256":REPO_PATH/"scripts/research/run_mr_deeper_dip_study.py",
        "spec_sha256":STUDY_PATH/"research_spec_frozen.json",
        "dependency_manifest_sha256":STUDY_PATH/"source_code_manifest.json",
        "input_manifest_sha256":STUDY_PATH/"data"/strategy_str/"input_manifest.json"}
    for key_str,path_obj in path_dict.items():
        if complete_dict[key_str]!=sha256_file(path_obj):
            raise AssertionError("Completed cell fingerprint mismatch: "+key_str)
    dependency_dict=json.loads((STUDY_PATH/"source_code_manifest.json").read_text())
    for name_str,digest_str in dependency_dict.items():
        if sha256_file(REPO_PATH/name_str)!=digest_str:
            raise AssertionError("Dependency drift: "+name_str)


def audit_dv2_observed_history():
    """Tag saved decisions; no indicators/signals/order states are recomputed."""
    from scripts.research.run_mr_deeper_dip_fast import freeze_frame_metadata
    native_df, _, _ = load_inputs("dv2")
    exact_df, _, _ = load_inputs("hpi235")
    freeze_frame_metadata(native_df)
    freeze_frame_metadata(exact_df)
    field_set={"Open","High","Low","Close"}
    native_bool_df=native_df.loc[:,native_df.columns.get_level_values(1).isin(field_set)].notna().T.groupby(level=0).all().T
    exact_bool_df=exact_df.loc[:,exact_df.columns.get_level_values(1).isin(field_set)].notna().T.groupby(level=0).all().T
    common_list=sorted(set(native_bool_df.columns)&set(exact_bool_df.columns))
    native_only_df=native_bool_df[common_list]&~exact_bool_df[common_list]
    # *** CRITICAL*** Retrospective input audit only; prior200 count at T
    # includes no observation after T and never affects any strategy decision.
    prior_count_df=native_only_df.rolling(200,min_periods=1).sum()
    last_index_dict={}
    index_vec=np.arange(len(native_only_df))
    for asset_str in common_list:
        last_index_dict[asset_str]=np.maximum.accumulate(
            np.where(native_only_df[asset_str].to_numpy(),index_vec,-1))
    entry_list, transaction_list, summary_list=[],[],[]
    for layer_str in ("central","stress"):
        for depth_float in DEPTH_TUPLE:
            policy_str=policy_name(depth_float)
            path_obj=STUDY_PATH/"runs"/"dv2"/f"{layer_str}_{policy_str}"
            _,entry_df,transaction_df,_,_=read_cell(path_obj)
            cell_entry_list=[]
            for order_row in entry_df.itertuples(index=False):
                known_bool=order_row.asset in common_list
                decision_int=native_only_df.index.get_loc(order_row.decision_date)
                last_int=last_index_dict[order_row.asset][decision_int] if known_bool else -1
                record_dict={
                    "layer":layer_str,"policy":policy_str,"date":order_row.date,
                    "decision_date":order_row.decision_date,"asset":order_row.asset,
                    "order_id":order_row.order_id,"status":order_row.status,
                    "reference_symbol_known":known_bool,
                    "decision_native_only":bool(native_only_df.loc[order_row.decision_date,order_row.asset]) if known_bool else None,
                    "prior200_native_only_count":int(prior_count_df.loc[order_row.decision_date,order_row.asset]) if known_bool else None,
                    "sessions_since_last_native_only":int(decision_int-last_int) if last_int>=0 else None,
                    "execution_native_only":bool(native_only_df.loc[order_row.date,order_row.asset]) if known_bool else None,
                }
                cell_entry_list.append(record_dict)
            tagged_df=pd.DataFrame(cell_entry_list)
            entry_list.extend(cell_entry_list)
            cell_transaction_list=[]
            for trade_row in transaction_df.itertuples(index=False):
                known_bool=trade_row.asset in common_list
                cell_transaction_list.append({
                    "layer":layer_str,"policy":policy_str,"date":trade_row.bar,
                    "asset":trade_row.asset,"order_id":trade_row.order_id,
                    "side":"entry" if trade_row.amount>0 else "exit",
                    "reference_symbol_known":known_bool,
                    "exact_observed_ohlc":bool(exact_bool_df.loc[trade_row.bar,trade_row.asset]) if known_bool else None,
                    "native_only_ohlc":bool(native_only_df.loc[trade_row.bar,trade_row.asset]) if known_bool else None,
                    "native_forced_liquidation":trade_row.order_id==-1,
                })
            tagged_tx_df=pd.DataFrame(cell_transaction_list)
            transaction_list.extend(cell_transaction_list)
            history_mask=tagged_df["prior200_native_only_count"].gt(0)
            summary_list.append({
                "layer":layer_str,"policy":policy_str,"selected":len(tagged_df),
                "selected_prior200_native_only":int(history_mask.sum()),
                "filled_prior200_native_only":int((history_mask&tagged_df["status"].eq("filled")).sum()),
                "selected_unknown_reference":int((~tagged_df["reference_symbol_known"]).sum()),
                "executions_native_only_ohlc":int(tagged_tx_df["native_only_ohlc"].eq(True).sum()),
                "executions_without_exact_ohlc":int(tagged_tx_df["exact_observed_ohlc"].eq(False).sum()),
                "forced_liquidations":int(tagged_tx_df["native_forced_liquidation"].sum()),
            })
    pd.DataFrame(entry_list).to_csv(TABLE_PATH/"dv2_input_exposure_orders.csv",index=False)
    pd.DataFrame(transaction_list).to_csv(TABLE_PATH/"dv2_input_exposure_transactions.csv",index=False)
    pd.DataFrame(summary_list).to_csv(TABLE_PATH/"dv2_input_exposure_summary.csv",index=False)



def summarize_entry_paths():
    path_list=[]
    for strategy_str in STRATEGY_TUPLE:
        baseline_df=pd.read_csv(STUDY_PATH/"runs"/strategy_str/"central_moo/entries.csv",
                                float_precision="round_trip")
        base_keys=set(zip(baseline_df.loc[baseline_df["status"].eq("filled"),"date"],
                          baseline_df.loc[baseline_df["status"].eq("filled"),"asset"]))
        for depth_float in DEPTH_TUPLE:
            policy_str=policy_name(depth_float)
            path_obj=STUDY_PATH/"runs"/strategy_str/f"central_{policy_str}"
            entry_df=pd.read_csv(path_obj/"entries.csv",float_precision="round_trip")
            daily_df=pd.read_csv(path_obj/"daily.csv",float_precision="round_trip")
            fill_df=entry_df.loc[entry_df["status"].eq("filled")]
            arm_keys=set(zip(fill_df["date"],fill_df["asset"]))
            if len(arm_keys)!=len(fill_df):
                raise AssertionError("Multiple entry trades for one asset/day.")
            path_list.append({"strategy":strategy_str,"policy":policy_str,
                "actual_fills":len(arm_keys),"same_asset_date_as_moo":len(arm_keys&base_keys),
                "different_asset_date_from_moo":len(arm_keys-base_keys),
                "end_of_day_slot_cap_frequency":float(daily_df["positions"].eq(
                    5 if strategy_str=="sector" else 10).mean())})
    pd.DataFrame(path_list).to_csv(TABLE_PATH/"entry_path_comparison.csv",index=False)



def analyze():
    TABLE_PATH.mkdir(exist_ok=True)
    CHART_PATH.mkdir(exist_ok=True)
    summary_list, period_list, year_list, inference_list = [], [], [], []
    paired_list, paired_summary_list, capacity_list, audit_list = [], [], [], []
    daily_return_dict, benchmark_dict, curve_dict = {}, {}, {}
    common_calendar_idx = None
    common_benchmark_ser = None
    for strategy_str in STRATEGY_TUPLE:
        pricing_df, universe_df, calendar_idx = load_inputs(strategy_str)
        from scripts.research.run_mr_deeper_dip_fast import freeze_frame_metadata
        freeze_frame_metadata(pricing_df)
        benchmark_str = "$SPXTR" if strategy_str=="hpi235" else "$SPX"
        # *** CRITICAL*** Benchmark ratios use previous close only and no padding.
        benchmark_ser = pricing_df[(benchmark_str,"Close")].pct_change(fill_method=None).loc[calendar_idx]
        if common_calendar_idx is None:
            common_calendar_idx, common_benchmark_ser = calendar_idx, benchmark_ser
        else:
            if not calendar_idx.equals(common_calendar_idx):
                raise AssertionError("Cross-strategy calendar mismatch.")
            pd.testing.assert_series_equal(benchmark_ser, common_benchmark_ser,
                                           check_names=False, check_freq=False)
        benchmark_dict[strategy_str] = benchmark_ser
        benchmark_metrics_dict = performance(benchmark_ser)
        write_json(TABLE_PATH/f"{strategy_str}_benchmark.json", benchmark_metrics_dict)
        baseline_trade_df = None
        baseline_entry_df = None
        for layer_str in ("central", "stress"):
            for depth_float in DEPTH_TUPLE:
                policy_str = policy_name(depth_float)
                cell_path = STUDY_PATH/"runs"/strategy_str/f"{layer_str}_{policy_str}"
                daily_df, entry_df, transaction_df, dividend_df, friction_df = read_cell(cell_path)
                if not daily_df.index.equals(calendar_idx):
                    raise AssertionError("Calendar mismatch.")
                verify_complete_fingerprints(strategy_str, cell_path)
                verify_execution_contract(strategy_str, layer_str, depth_float, entry_df,
                                          transaction_df, friction_df, pricing_df)
                cash_error_float = reconcile_cash(daily_df, transaction_df, dividend_df, friction_df)
                mark_error_float = reconcile_daily_marks(daily_df, transaction_df, pricing_df)
                dividend_event_count_int = verify_dividend_entitlements(
                    calendar_idx, pricing_df, transaction_df, dividend_df)
                return_ser = nav_returns(daily_df["nav"])
                trade_df = make_trade_table(transaction_df, dividend_df, friction_df, pricing_df)
                total_error_float = abs(float(trade_df["net_pnl"].sum())-(daily_df["nav"].iloc[-1]-100000.))
                if total_error_float > .00001:
                    raise AssertionError(f"Trade/NAV attribution failed {total_error_float}")
                trade_df.to_csv(cell_path/"all_in_trades.csv", index=False)
                closed_df = trade_df.loc[trade_df["closed"]]
                pnl_loss_ser = closed_df.loc[closed_df["net_return"]<0., "net_return"]
                monthly_ser = (1.+return_ser).resample("ME").prod()-1.
                benchmark_monthly_ser = (1.+benchmark_ser).resample("ME").prod()-1.
                beta_float = float(return_ser.cov(benchmark_ser)/benchmark_ser.var())
                filled_mask = entry_df["status"].eq("filled")
                metric_dict = {
                    "strategy": strategy_str, "layer": layer_str, "policy": policy_str,
                    **performance(return_ser),
                    "final_nav": float(daily_df["nav"].iloc[-1]),
                    "mean_exposure": float((daily_df["invested"]/daily_df["nav"]).mean()),
                    "mean_positions": float(daily_df["positions"].mean()),
                    "negative_cash_days": int((daily_df["cash"]<-.01).sum()),
                    "mean_borrowed_weight": float((-daily_df["cash"]/daily_df["nav"]).clip(lower=0.).mean()),
                    "min_cash_weight": float((daily_df["cash"]/daily_df["nav"]).min()),
                    "orders": len(entry_df), "fills": int(filled_mask.sum()),
                    "fill_rate": float(filled_mask.mean()), "closed_trades": len(closed_df),
                    "open_trades": int((~trade_df["closed"]).sum()),
                    "native_forced_liquidations": int(transaction_df["order_id"].eq(-1).sum()),
                    "closed_trade_net_pnl": float(closed_df["net_pnl"].sum()),
                    "terminal_trade_net_pnl": float(trade_df.loc[~trade_df["closed"],"net_pnl"].sum()),
                    "terminal_mark_value": float(trade_df["terminal_mark_value"].sum()),
                    "mean_trade_return": float(closed_df["net_return"].mean()),
                    "median_trade_return": float(closed_df["net_return"].median()),
                    "win_rate": float(closed_df["net_return"].gt(0.).mean()),
                    "worst_trade": float(closed_df["net_return"].min()),
                    "mean_loss": float(pnl_loss_ser.mean()),
                    "trade_cvar5": float(closed_df.loc[
                        closed_df["net_return"]<=closed_df["net_return"].quantile(.05), "net_return"].mean()),
                    "mean_duration": float(closed_df["duration_sessions"].mean()),
                    "daily_correlation": float(return_ser.corr(benchmark_ser)),
                    "monthly_correlation": float(monthly_ser.corr(benchmark_monthly_ser)),
                    "beta": beta_float,
                    "annual_turnover": float(transaction_df["total_value"].abs().sum()/daily_df["nav"].mean()*252./len(daily_df)),
                    "commission_total": float(transaction_df["commission"].sum()),
                    "cash_friction_total": float(friction_df["research_friction"].sum()),
                    "dividend_total": float(dividend_df["net_dividend_cash_float"].sum()),
                    "dividend_events_independently_verified": dividend_event_count_int,
                    "cash_error_max": cash_error_float, "trade_nav_error": total_error_float,
                    "daily_mark_error": mark_error_float,
                    "zero_volume_selected": int(entry_df["volume"].le(0.).sum()),
                    "zero_volume_filled": int((filled_mask & entry_df["volume"].le(0.)).sum()),
                }
                summary_list.append(metric_dict)
                daily_return_dict[(strategy_str,layer_str,policy_str)] = return_ser
                if layer_str=="central":
                    curve_dict[(strategy_str,policy_str)] = daily_df["nav"]/100000.
                for period_str, start_str, end_str in PERIOD_TUPLE:
                    period_ser = return_ser.loc[start_str:end_str]
                    period_list.append({"strategy":strategy_str,"layer":layer_str,"policy":policy_str,
                                        "period":period_str,**performance(period_ser)})
                for year_int, year_return_ser in return_ser.groupby(return_ser.index.year):
                    year_list.append({"strategy":strategy_str,"layer":layer_str,"policy":policy_str,
                                      "year":int(year_int),"return":float((1.+year_return_ser).prod()-1.)})
                if layer_str=="central":
                    # *** CRITICAL*** ADV63 is read at decision close T, never
                    # at T+1. Rolling turnover includes only past/current T data.
                    for asset_str, asset_entry_df in entry_df.groupby("asset"):
                        adv_ser = pricing_df[(asset_str,"Turnover")].rolling(63,min_periods=63).mean()
                        for order_row in asset_entry_df.itertuples(index=False):
                            adv_float = float(adv_ser.loc[order_row.decision_date])
                            notional_float = float(order_row.shares*order_row.open)
                            capacity_list.append({
                                "strategy":strategy_str,"policy":policy_str,"date":order_row.date,
                                "asset":asset_str,"status":order_row.status,
                                "notional_at_open":notional_float,"adv63":adv_float,
                                "participation":notional_float/adv_float if adv_float>0 else np.nan})
                if layer_str=="central" and depth_float is None:
                    baseline_trade_df, baseline_entry_df = trade_df, entry_df
                    native_df = pd.read_csv(float_precision="round_trip", filepath_or_buffer=STUDY_PATH/"runs"/strategy_str/"native_transactions.csv", parse_dates=["bar"])
                    pd.testing.assert_frame_equal(native_df.drop(columns="order_id"),
                                                  transaction_df.drop(columns="order_id"),check_dtype=False)
                    audit_list.append({"strategy":strategy_str,"native_transaction_parity":True,
                                       "native_dividend_event_parity":True,
                                       "dividend_events":dividend_event_count_int})
        paired_df = paired_opportunities(strategy_str, pricing_df, baseline_entry_df, baseline_trade_df)
        paired_list.append(paired_df)
        for policy_str, policy_df in paired_df.groupby("policy",sort=False):
            filled_df = policy_df.loc[policy_df["filled"]]
            paired_summary_list.append({
                "strategy":strategy_str,"policy":policy_str,"opportunities":len(policy_df),
                "fills":len(filled_df),"fill_rate":len(filled_df)/len(policy_df),
                "closed_opportunities":int(policy_df["closed"].sum()),
                "terminal_opportunities":int((~policy_df["closed"]).sum()),
                "closed_delta_pnl":float(policy_df.loc[policy_df["closed"],"delta_net_pnl"].sum()),
                "terminal_delta_pnl":float(policy_df.loc[~policy_df["closed"],"delta_net_pnl"].sum()),
                "baseline_pnl":float(policy_df["baseline_net_pnl"].sum()),
                "limit_pnl":float(policy_df["limit_net_pnl"].sum()),
                "delta_pnl":float(policy_df["delta_net_pnl"].sum()),
                "mean_delta_per_opportunity":float(policy_df["delta_net_pnl"].mean()),
                "mean_baseline_return_all_opportunities":float(policy_df["baseline_return"].mean()),
                "mean_limit_return_all_opportunities":float(policy_df["limit_return_or_zero"].mean()),
                "net_fill_price_improvement":float(filled_df["delta_net_pnl"].sum()),
                "mean_limit_return_filled":float(filled_df["limit_return_or_zero"].mean()),
                "mean_baseline_return_same_filled":float(filled_df["baseline_return"].mean()),
                "missed_baseline_pnl":float(policy_df["missed_baseline_net_pnl"].sum()),
                "gross_fill_price_improvement":float(policy_df["gross_price_improvement_on_fill"].sum()),
            })
        del pricing_df, universe_df

    audit_dv2_observed_history()
    summarize_entry_paths()
    timing_list=[]
    for paired_df in paired_list:
        timing_df=paired_df.loc[paired_df["policy"].eq("limit_0.5pct")]
        assert np.allclose((1.+timing_df["overnight_price_return"])*(1.+timing_df["open_to_exit_price_return"]),1.+timing_df["close_to_exit_price_return"])
        timing_list.append({"strategy":timing_df["strategy"].iloc[0], "opportunities":len(timing_df),
            **{field_str:float(timing_df[field_str].mean()) for field_str in ("overnight_price_return","open_to_exit_price_return","close_to_exit_price_return")}})
    pd.DataFrame(timing_list).to_csv(TABLE_PATH/"timing_attribution.csv",index=False)
    summary_df = pd.DataFrame(summary_list)
    for strategy_str in STRATEGY_TUPLE:
        baseline_ser = daily_return_dict[(strategy_str,"central","moo")]
        for depth_float in DEPTH_TUPLE[1:]:
            policy_str = policy_name(depth_float)
            delta_ser = daily_return_dict[(strategy_str,"central",policy_str)]-baseline_ser
            model_obj = sm.OLS(delta_ser.to_numpy(),np.ones((len(delta_ser),1))).fit(
                cov_type="HAC",cov_kwds={"maxlags":5})
            interval_mat = model_obj.conf_int()
            inference_list.append({
                "strategy":strategy_str,"policy":policy_str,
                "mean_daily_delta":float(model_obj.params[0]),
                "annualized_arithmetic_delta":float(model_obj.params[0]*252),
                "hac_t":float(model_obj.tvalues[0]),"p_two_sided":float(model_obj.pvalues[0]),
                "ci95_low_annual":float(interval_mat[0,0]*252),
                "ci95_high_annual":float(interval_mat[0,1]*252)})
    inference_df = pd.DataFrame(inference_list)
    inference_df["p_holm_9"] = multipletests(inference_df["p_two_sided"],method="holm")[1]
    summary_df.to_csv(TABLE_PATH/"summary.csv",index=False)
    pd.DataFrame(period_list).to_csv(TABLE_PATH/"subperiods.csv",index=False)
    pd.DataFrame(year_list).to_csv(TABLE_PATH/"annual_returns.csv",index=False)
    inference_df.to_csv(TABLE_PATH/"inference.csv",index=False)
    pd.concat(paired_list,ignore_index=True).to_csv(TABLE_PATH/"paired_opportunities.csv",index=False)
    pd.DataFrame(paired_summary_list).to_csv(TABLE_PATH/"paired_summary.csv",index=False)
    capacity_df = pd.DataFrame(capacity_list)
    capacity_df.to_csv(TABLE_PATH/"capacity_orders.csv",index=False)
    capacity_summary_list = []
    for (strategy_str,policy_str), group_df in capacity_df.groupby(["strategy","policy"]):
        participant_ser=group_df["participation"].dropna()
        capacity_summary_list.append({"strategy":strategy_str,"policy":policy_str,
            "p50":participant_ser.quantile(.5),"p90":participant_ser.quantile(.9),
            "p99":participant_ser.quantile(.99),"max":participant_ser.max(),
            "missing_adv":int((group_df["adv63"].isna()|group_df["adv63"].le(0.)).sum()),
            "missing_request_notional":int(group_df["notional_at_open"].isna().sum())})
    pd.DataFrame(capacity_summary_list).to_csv(TABLE_PATH/"capacity_summary.csv",index=False)
    pooled_df = capacity_df.groupby(["policy","date","asset"]).agg(
        notional=("notional_at_open",lambda value_ser:value_ser.sum(min_count=1)),
        notional_missing=("notional_at_open",lambda value_ser:int(value_ser.isna().sum())),adv_min=("adv63","min"),
        adv_max=("adv63","max"),sleeves=("strategy","nunique"),
        adv_missing=("adv63",lambda value_ser:int(value_ser.isna().sum())))
    pooled_df["adv_mismatch"] = ~np.isclose(pooled_df["adv_min"],pooled_df["adv_max"],
                                           rtol=1e-10,atol=1e-6,equal_nan=True)
    # Conservative REQUESTED entry-demand proxy: use smaller known ADV on
    # native-loader disagreement; disclose every mismatch, never silently first.
    pooled_df["participation"]=(pooled_df["notional"]/pooled_df["adv_min"]).where(
        pooled_df["notional_missing"].eq(0)&pooled_df["adv_missing"].eq(0)&pooled_df["adv_min"].gt(0.))
    pooled_df.to_csv(TABLE_PATH/"capacity_pooled_demand.csv")
    write_json(TABLE_PATH/"accounting_audit.json",audit_list)
    pd.DataFrame({"/".join(key_tuple):value_ser for key_tuple,value_ser in daily_return_dict.items()}).to_csv(TABLE_PATH/"daily_returns.csv")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    color_list=["#334155","#2563eb","#d97706","#dc2626"]
    figure_obj, axes_mat=plt.subplots(3,2,figsize=(14,11),sharex=True)
    for row_int,strategy_str in enumerate(STRATEGY_TUPLE):
        for color_str,depth_float in zip(color_list,DEPTH_TUPLE):
            policy_str=policy_name(depth_float)
            curve_ser=curve_dict[(strategy_str,policy_str)]
            axes_mat[row_int,0].plot(curve_ser.index,curve_ser,label=DISPLAY_POLICY_DICT[policy_str],color=color_str)
            drawdown_ser=curve_ser/curve_ser.cummax().clip(lower=1.)-1.
            axes_mat[row_int,1].plot(drawdown_ser.index,drawdown_ser*100.,color=color_str)
        benchmark_curve_ser=(1.+benchmark_dict[strategy_str]).cumprod()
        axes_mat[row_int,0].plot(benchmark_curve_ser.index,benchmark_curve_ser,
                                "--",color="#94a3b8",label="SPX total return")
        axes_mat[row_int,0].set_yscale("log")
        axes_mat[row_int,0].set_ylabel(DISPLAY_NAME_DICT[strategy_str]+" | growth of $1")
        axes_mat[row_int,1].set_ylabel("Drawdown (%)")
        axes_mat[row_int,0].legend(fontsize=8,ncol=2)
    figure_obj.suptitle("Mean-reversion entry policies | 2010-01-05 to 2026-09-14\nCentral costs; original exits; ideal daily-OHLC limit fills")
    figure_obj.tight_layout()
    figure_obj.savefig(CHART_PATH/"equity_drawdown.png",dpi=155)
    plt.close(figure_obj)

    figure_obj,axes_arr=plt.subplots(1,3,figsize=(14,4.5))
    for axis_obj,strategy_str in zip(axes_arr,STRATEGY_TUPLE):
        group_df=summary_df[(summary_df["strategy"]==strategy_str)&(summary_df["layer"]=="central")]
        index_arr=np.arange(4)
        axis_obj.bar(index_arr-.18,group_df["fill_rate"]*100.,width=.36,label="Fill rate")
        axis_obj.bar(index_arr+.18,group_df["mean_exposure"]*100.,width=.36,label="Mean exposure")
        axis_obj.set_xticks(index_arr,["MOO","-0.5%","-1%","-2%"])
        axis_obj.set_title(DISPLAY_NAME_DICT[strategy_str])
        axis_obj.set_ylim(0,105)
        axis_obj.set_ylabel("%")
        axis_obj.legend(fontsize=8)
    figure_obj.suptitle("Fill frequency and capital use | full common period | central costs")
    figure_obj.tight_layout()
    figure_obj.savefig(CHART_PATH/"fills_exposure.png",dpi=160)
    plt.close(figure_obj)

    figure_obj,axes_arr=plt.subplots(1,3,figsize=(14,4.5))
    for axis_obj,strategy_str in zip(axes_arr,STRATEGY_TUPLE):
        for color_str,depth_float in zip(color_list,DEPTH_TUPLE):
            policy_str=policy_name(depth_float)
            return_ser=daily_return_dict[(strategy_str,"central",policy_str)]
            # *** CRITICAL*** Trailing126 observed daily returns, no future window.
            rolling_ser=return_ser.rolling(126,min_periods=126).corr(benchmark_dict[strategy_str])
            axis_obj.plot(rolling_ser.index,rolling_ser,label=DISPLAY_POLICY_DICT[policy_str],color=color_str,linewidth=.8)
        axis_obj.set_title(DISPLAY_NAME_DICT[strategy_str])
        axis_obj.set_ylim(-1,1)
    figure_obj.suptitle("Trailing 126-session market correlation | central costs; no forward window")
    axes_arr[0].legend(fontsize=7,ncol=2)
    figure_obj.tight_layout()
    figure_obj.savefig(CHART_PATH/"rolling_correlation.png",dpi=150)
    plt.close(figure_obj)

    figure_obj,axes_arr=plt.subplots(1,3,figsize=(14,4.5))
    for axis_obj,strategy_str in zip(axes_arr,STRATEGY_TUPLE):
        for layer_str,style_str in (("central","-o"),("stress","--s")):
            group_df=summary_df[(summary_df["strategy"]==strategy_str)&(summary_df["layer"]==layer_str)]
            axis_obj.plot(np.arange(4),group_df["cagr"]*100.,style_str,label=layer_str)
        axis_obj.set_xticks(np.arange(4),["MOO","-0.5%","-1%","-2%"])
        axis_obj.set_title(DISPLAY_NAME_DICT[strategy_str])
        axis_obj.set_ylabel("CAGR (%)")
        axis_obj.legend()
    figure_obj.suptitle("Cost + fill sensitivity | stress adds 10bp/side and requires 5bp limit penetration")
    figure_obj.tight_layout()
    figure_obj.savefig(CHART_PATH/"cost_sensitivity.png",dpi=160)
    plt.close(figure_obj)
    print(summary_df[["strategy","layer","policy","cagr","sharpe","max_drawdown","fill_rate","mean_exposure"]].to_string(index=False))
    print("PASS: 24 cells, cash/NAV/trade/dividend checks, paired diagnostics and figures.")


if __name__=="__main__":
    analyze()
