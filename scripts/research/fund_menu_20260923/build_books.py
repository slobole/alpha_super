"""Phase 3: build every product from the FROZEN spec, then evaluate and gate it.

Order of operations (the spec hash is checked first, so a result can never feed
back into the rules that produced it):

1. read frozen_spec.yaml and verify its sha256 against frozen_spec.sha256;
2. estimate the weekly covariance on the exact window (risk only);
3. for each product, pick the return-engine share s that meets its volatility
   target on the two-bucket capital template (templates.py), round to 1%;
   then apply any post-freeze owner amendment from amendments.yaml (checked,
   logged in the ledger, and labelled in the report; the frozen spec is untouched);
4. build books as sums of independently compounded pods, reset to target weights
   each year-end (product policy; drift is the sensitivity), on the exact window
   and on a long window where the BTAL-based TAA is replaced by its 2008 sibling;
5. metrics, crises, sub-periods, rolling windows, bootstrap, perturbation,
   estimation split, alternative constructions, hedge overlays, cost/financing
   stress, cash-interest upside, and the current ladder books on the same window;
6. pre-declared gates.

Outputs: results/research/portfolio/fund_product_menu_20260923/books/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import construction  # noqa: E402
import evaluation  # noqa: E402
import templates  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent / "frozen_spec.yaml"
SPEC_HASH_PATH = Path(__file__).resolve().parent / "frozen_spec.sha256"
AMENDMENT_PATH = Path(__file__).resolve().parent / "amendments.yaml"
INVENTORY_DIR_PATH = common.STUDY_DIR_PATH / "inventory"
BOOK_DIR_PATH = common.STUDY_DIR_PATH / "books"
LEDGER_PATH = common.STUDY_DIR_PATH / "experiment_ledger.jsonl"


def load_frozen_spec_dict() -> dict:
    spec_bytes = SPEC_PATH.read_bytes()
    actual_sha_str = hashlib.sha256(spec_bytes).hexdigest()
    frozen_sha_str = SPEC_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if actual_sha_str != frozen_sha_str:
        raise RuntimeError("frozen_spec.yaml changed after it was frozen; refusing to build.")
    return yaml.safe_load(spec_bytes)


def load_amendment_by_product_dict() -> dict[str, dict]:
    """Post-freeze owner decisions (amendments.yaml), keyed by product. The frozen spec is never edited."""
    if not AMENDMENT_PATH.exists():
        return {}
    amendment_list = yaml.safe_load(AMENDMENT_PATH.read_text(encoding="utf-8"))["amendments"]
    return {a["product_id_str"]: a for a in amendment_list}


def drop_amended_ser(weight_ser: pd.Series, amendment_dict: dict | None) -> pd.Series:
    """Remove the amendment's sleeves and scale the remaining pods up pro rata (the sum is preserved)."""
    if not amendment_dict:
        return weight_ser
    kept_ser = weight_ser.drop(labels=[a for a in amendment_dict["drop_alias_list"] if a in weight_ser.index])
    return kept_ser / kept_ser.sum() * weight_ser.sum()


def amended_final_weight_ser(raw_ser: pd.Series, amendment_dict: dict, step_float: float) -> pd.Series:
    """The amendment's stated weights, checked against its own rule before use."""
    final_ser = pd.Series(amendment_dict["final_weight_dict"], dtype=float)
    pro_rata_ser = raw_ser / raw_ser.sum()
    if set(final_ser.index) != set(pro_rata_ser.index):
        raise RuntimeError(f"{amendment_dict['amendment_id_str']}: final pods differ from the template pods minus the dropped ones.")
    if abs(final_ser.sum() - 1.0) > 1e-9 or (final_ser - pro_rata_ser.reindex(final_ser.index)).abs().max() > step_float + 1e-9:
        raise RuntimeError(f"{amendment_dict['amendment_id_str']}: final weights break the stated pro-rata rounding rule.")
    return final_ser.reindex(pro_rata_ser.sort_values(ascending=False).index)


def load_return_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    sleeve_return_df = pd.read_csv(INVENTORY_DIR_PATH / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    benchmark_return_df = pd.read_csv(INVENTORY_DIR_PATH / "benchmark_returns.csv.gz", index_col=0, parse_dates=True)
    return sleeve_return_df, benchmark_return_df


def window_start_ts(sleeve_return_df: pd.DataFrame, alias_list: list[str]) -> pd.Timestamp:
    """First session on which every listed sleeve has a realised return."""
    return max(sleeve_return_df[alias_str].first_valid_index() for alias_str in alias_list)


def previous_session_ts(index: pd.DatetimeIndex, session_ts: pd.Timestamp) -> pd.Timestamp:
    return index[index.get_loc(session_ts) - 1]


def weekly_covariance_df(return_df: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance of Friday-to-Friday compounded returns.

    Weekly sampling absorbs one-day lead/lag between sleeves that fill at the
    open and sleeves that fill at the close, which daily data would read as
    lower correlation than the books actually carry. Only sleeves with complete
    data in the window enter.
    """
    complete_return_df = return_df.dropna(axis=1, how="any")
    weekly_return_df = (1.0 + complete_return_df).resample("W-FRI").prod() - 1.0
    return weekly_return_df.cov() * 52.0


def product_weight_ser(spec_dict: dict, product_dict: dict, covariance_df: pd.DataFrame, **template_kwargs) -> tuple[float, pd.Series]:
    if "target_volatility_float" in product_dict:
        share_float, raw_ser = templates.solve_share_for_target_volatility(
            spec_dict, product_dict["line_str"], covariance_df, float(product_dict["target_volatility_float"]), **template_kwargs
        )
    else:
        share_float = float(product_dict["return_share_float"])
        raw_ser = templates.template_weight_ser(spec_dict, product_dict["line_str"], share_float, **template_kwargs)
    return share_float, raw_ser


def full_metric_dict(return_ser: pd.Series, benchmark_return_df: pd.DataFrame, base_ts: pd.Timestamp) -> dict:
    return common.metric_dict(return_ser, benchmark_return_df["SPXTR"], benchmark_return_df["TBILL"], base_ts)


def rolling_window_dict(return_ser: pd.Series) -> dict:
    """Worst and loss-probability over rolling 12- and 36-month holding periods (daily steps)."""
    nav_ser = common.nav_from_return_ser(return_ser)
    out_dict = {}
    for label_str, session_int in (("12m", 252), ("36m", 756)):
        holding_return_ser = (nav_ser / nav_ser.shift(session_int) - 1.0).dropna()
        out_dict[f"worst_{label_str}_float"] = float(holding_return_ser.min()) if len(holding_return_ser) else np.nan
        out_dict[f"share_{label_str}_negative_float"] = float((holding_return_ser < 0).mean()) if len(holding_return_ser) else np.nan
    return out_dict


def crisis_table_df(return_by_series_dict: dict[str, pd.Series], episode_list: list[dict]) -> pd.DataFrame:
    row_list = []
    for episode_dict in episode_list:
        row_dict = {"episode_str": episode_dict["label_str"], "start_str": episode_dict["peak_date_str"], "end_str": episode_dict["trough_date_str"]}
        for series_str, return_ser in return_by_series_dict.items():
            covered_bool = return_ser.index[0] <= pd.Timestamp(episode_dict["peak_date_str"]) + pd.Timedelta(days=1)
            row_dict[series_str] = (
                common.window_return_float(return_ser, episode_dict["peak_date_str"], episode_dict["trough_date_str"]) if covered_bool else np.nan
            )
        row_list.append(row_dict)
    return pd.DataFrame(row_list)


def labelled_episode_list(spx_return_ser: pd.Series, named_window_list: list[dict]) -> list[dict]:
    episode_list = []
    for episode_dict in common.equity_drawdown_episode_list(spx_return_ser, threshold_float=-0.10):
        peak_ts = pd.Timestamp(episode_dict["peak_date_str"])
        trough_ts = pd.Timestamp(episode_dict["trough_date_str"])
        episode_list.append({
            **episode_dict,
            "label_str": f"S&P 500 −{abs(episode_dict['spx_drawdown_float']) * 100:.0f}%: {peak_ts:%b %Y} to {trough_ts:%b %Y}",
            "kind_str": "equity_drawdown",
        })
    for window_dict in named_window_list:
        episode_list.append({"peak_date_str": window_dict["start_str"], "trough_date_str": window_dict["end_str"],
                             "label_str": window_dict["label_str"], "kind_str": "named_window"})
    return sorted(episode_list, key=lambda d: d["peak_date_str"])


def subperiod_df(return_by_series_dict: dict[str, pd.Series], benchmark_return_df: pd.DataFrame, part_count_int: int = 3) -> pd.DataFrame:
    any_ser = next(iter(return_by_series_dict.values()))
    edge_arr = np.linspace(0, len(any_ser), part_count_int + 1).astype(int)
    row_list = []
    for part_int in range(part_count_int):
        part_index = any_ser.index[edge_arr[part_int]: edge_arr[part_int + 1]]
        base_ts = any_ser.index[edge_arr[part_int] - 1] if edge_arr[part_int] > 0 else part_index[0] - pd.Timedelta(days=1)
        for series_str, return_ser in return_by_series_dict.items():
            metric = full_metric_dict(return_ser.loc[part_index], benchmark_return_df, base_ts)
            row_list.append({
                "part_int": part_int + 1, "start_str": part_index[0].date().isoformat(), "end_str": part_index[-1].date().isoformat(),
                "series_str": series_str, "cagr_float": metric["cagr_float"], "excess_cagr_float": metric["cagr_float"] - metric["tbill_cagr_float"],
                "volatility_float": metric["volatility_float"], "sharpe_rf0_float": metric["sharpe_rf0_float"], "max_drawdown_float": metric["max_drawdown_float"],
            })
    return pd.DataFrame(row_list)


def calendar_year_df(return_by_series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        {series_str: (1.0 + ser).resample("YE").prod() - 1.0 for series_str, ser in return_by_series_dict.items()}
    ).rename(index=lambda ts: ts.year)


def dirichlet_around(rng: np.random.Generator, base_dict: dict[str, float], concentration_float: float) -> dict[str, float]:
    key_list = list(base_dict)
    if len(key_list) == 1:
        return dict(base_dict)
    base_arr = np.array([base_dict[k] for k in key_list], dtype=float)
    base_arr = base_arr / base_arr.sum()
    return dict(zip(key_list, rng.dirichlet(base_arr * concentration_float * len(key_list))))


def main() -> int:  # noqa: C901 - one linear study script, kept in one place on purpose
    spec_dict = load_frozen_spec_dict()
    BOOK_DIR_PATH.mkdir(parents=True, exist_ok=True)
    sleeve_return_df, benchmark_return_df = load_return_inputs()
    end_ts = pd.Timestamp(spec_dict["end_date_str"])
    sleeve_return_df = sleeve_return_df.loc[:end_ts]
    benchmark_return_df = benchmark_return_df.loc[:end_ts]
    rounding_step_float = float(spec_dict["rounding_step_float"])
    swap_dict = dict(spec_dict["long_window_alias_swap"])
    product_list = spec_dict["products"]
    product_by_id_dict = {p["product_id_str"]: p for p in product_list}
    amendment_by_product_dict = load_amendment_by_product_dict()
    if amendment_by_product_dict:
        amendment_bytes = AMENDMENT_PATH.read_bytes()
        with LEDGER_PATH.open("a", encoding="utf-8") as ledger_file_obj:
            ledger_file_obj.write(json.dumps({
                "event_str": "post_freeze_amendments_applied",
                "recorded_at_utc_str": pd.Timestamp.now(tz="UTC").isoformat(),
                "amendments_sha256_str": hashlib.sha256(amendment_bytes).hexdigest(),
                "amendment_id_list": [a["amendment_id_str"] for a in amendment_by_product_dict.values()],
                "note_str": "owner decisions taken after seeing results; frozen_spec.yaml unchanged",
            }) + "\n")

    # Windows. The exact window starts when every sleeve any product can hold has a return.
    exact_alias_set: set[str] = set()
    for line_dict in spec_dict["lines"].values():
        for engine_str in line_dict["return_engines"]:
            exact_alias_set.add(spec_dict["engines"][engine_str]["primary_str"])
            if spec_dict["engines"][engine_str].get("secondary_str"):
                exact_alias_set.add(spec_dict["engines"][engine_str]["secondary_str"])
        exact_alias_set.update(line_dict["stabilizer_capital"])
    long_alias_set = {swap_dict.get(a, a) for a in exact_alias_set}
    exact_start_ts = window_start_ts(sleeve_return_df, sorted(exact_alias_set))
    long_start_ts = window_start_ts(sleeve_return_df, sorted(long_alias_set))
    exact_index = sleeve_return_df.loc[exact_start_ts:].index
    long_index = sleeve_return_df.loc[long_start_ts:].index
    exact_base_ts = previous_session_ts(sleeve_return_df.index, exact_start_ts)
    long_base_ts = previous_session_ts(sleeve_return_df.index, long_start_ts)
    covariance_exact_df = weekly_covariance_df(sleeve_return_df.loc[exact_index, sorted(exact_alias_set)])
    exact_return_df = sleeve_return_df.loc[exact_index]
    long_return_df = sleeve_return_df.loc[long_index]

    # Products.
    product_result_dict: dict[str, dict] = {}
    exact_book_dict: dict[str, pd.Series] = {}
    drift_book_dict: dict[str, pd.Series] = {}
    long_book_dict: dict[str, pd.Series] = {}
    risk_frame_list, weight_row_list = [], []
    for product_dict in product_list:
        product_id_str = product_dict["product_id_str"]
        amendment_dict = amendment_by_product_dict.get(product_id_str)
        share_float, raw_ser = product_weight_ser(spec_dict, product_dict, covariance_exact_df)
        # *** CRITICAL*** a post-freeze amendment edits the template's output, never the frozen rules.
        raw_ser = drop_amended_ser(raw_ser, amendment_dict)
        weight_ser = (amended_final_weight_ser(raw_ser, amendment_dict, rounding_step_float) if amendment_dict
                      else construction.round_weight_ser(raw_ser, rounding_step_float))
        weight_dict = weight_ser.to_dict()
        annual_ser, annual_weight_df = common.book_return_ser(exact_return_df, weight_dict, "annual")
        drift_ser, drift_weight_df = common.book_return_ser(exact_return_df, weight_dict, "none")
        long_weight_dict = {}
        for alias_str, w in weight_dict.items():
            long_weight_dict[swap_dict.get(alias_str, alias_str)] = long_weight_dict.get(swap_dict.get(alias_str, alias_str), 0.0) + w
        long_ser, _ = common.book_return_ser(long_return_df, long_weight_dict, "annual")
        exact_book_dict[product_id_str] = annual_ser
        drift_book_dict[product_id_str] = drift_ser
        long_book_dict[product_id_str] = long_ser
        sigma_mat = covariance_exact_df.loc[weight_ser.index, weight_ser.index].to_numpy()
        ex_ante_share_ser = pd.Series(construction.risk_share_arr(sigma_mat, weight_ser.to_numpy()), index=weight_ser.index)
        realised_df = common.risk_share_df(exact_return_df, annual_weight_df, annual_ser)
        realised_df["product_id_str"] = product_id_str
        realised_df["ex_ante_risk_share_float"] = ex_ante_share_ser
        realised_df["drift_end_weight_float"] = drift_weight_df.iloc[-1]
        risk_frame_list.append(realised_df.reset_index().rename(columns={"index": "alias_str"}))
        for alias_str, w in weight_dict.items():
            weight_row_list.append({"product_id_str": product_id_str, "alias_str": alias_str, "weight_float": w,
                                    "raw_weight_float": float(raw_ser.get(alias_str, np.nan)),
                                    "long_window_alias_str": swap_dict.get(alias_str, alias_str)})
        product_result_dict[product_id_str] = {
            "return_share_float": share_float,
            "ex_ante_volatility_float": construction.book_volatility_float(covariance_exact_df, weight_ser),
            "weight_dict": weight_dict,
            "long_weight_dict": long_weight_dict,
            "amendment_id_str": amendment_dict["amendment_id_str"] if amendment_dict else None,
        }
    pd.DataFrame(weight_row_list).to_csv(BOOK_DIR_PATH / "product_weights.csv", index=False, float_format="%.6g")
    pd.concat(risk_frame_list).to_csv(BOOK_DIR_PATH / "product_risk_shares.csv", index=False, float_format="%.6g")

    # Current ladder books on the same exact window, as defined in portfolios/*.yaml (no reset).
    legacy_book_dict = {k: common.book_return_ser(exact_return_df, v, "none")[0] for k, v in spec_dict["legacy_books"].items()}
    benchmark_exact_dict = {"S&P 500 TR": benchmark_return_df.loc[exact_index, "SPXTR"], "60/40": benchmark_return_df.loc[exact_index, "SIXTY_FORTY"],
                            "T-bills": benchmark_return_df.loc[exact_index, "TBILL"]}
    benchmark_long_dict = {"S&P 500 TR": benchmark_return_df.loc[long_index, "SPXTR"], "60/40": benchmark_return_df.loc[long_index, "SIXTY_FORTY"],
                           "T-bills": benchmark_return_df.loc[long_index, "TBILL"]}

    metric_row_list = []
    for window_str, series_dict, base_ts in (
        ("exact_annual", exact_book_dict, exact_base_ts), ("exact_drift", drift_book_dict, exact_base_ts),
        ("exact_benchmark", benchmark_exact_dict, exact_base_ts), ("exact_legacy", legacy_book_dict, exact_base_ts),
        ("long_annual", long_book_dict, long_base_ts), ("long_benchmark", benchmark_long_dict, long_base_ts),
    ):
        for series_str, return_ser in series_dict.items():
            metric_row_list.append({"window_str": window_str, "series_str": series_str,
                                    **full_metric_dict(return_ser, benchmark_return_df, base_ts), **rolling_window_dict(return_ser)})
    metric_df = pd.DataFrame(metric_row_list)
    metric_df.to_csv(BOOK_DIR_PATH / "headline_metrics.csv", index=False, float_format="%.6g")

    product_corr_df = pd.DataFrame({**exact_book_dict, **legacy_book_dict, "S&P 500 TR": benchmark_exact_dict["S&P 500 TR"],
                                    "60/40": benchmark_exact_dict["60/40"]}).corr()
    product_corr_df.to_csv(BOOK_DIR_PATH / "product_correlations_daily.csv", float_format="%.4f")

    episode_list = labelled_episode_list(benchmark_return_df.loc[long_index, "SPXTR"], spec_dict["named_windows"])
    crisis_table_df({**long_book_dict, **benchmark_long_dict}, episode_list).to_csv(BOOK_DIR_PATH / "crisis_long_window.csv", index=False, float_format="%.6g")
    crisis_table_df({**exact_book_dict, **legacy_book_dict, **benchmark_exact_dict}, episode_list).to_csv(
        BOOK_DIR_PATH / "crisis_exact_window.csv", index=False, float_format="%.6g")
    subperiod_frame_df = subperiod_df({**exact_book_dict, **legacy_book_dict, "60/40": benchmark_exact_dict["60/40"],
                                       "S&P 500 TR": benchmark_exact_dict["S&P 500 TR"]}, benchmark_return_df)
    subperiod_frame_df.to_csv(BOOK_DIR_PATH / "subperiods_exact.csv", index=False, float_format="%.6g")
    calendar_year_df({**long_book_dict, **benchmark_long_dict}).to_csv(BOOK_DIR_PATH / "calendar_years_long.csv", float_format="%.6g")
    calendar_year_df({**exact_book_dict, **legacy_book_dict, **benchmark_exact_dict}).to_csv(BOOK_DIR_PATH / "calendar_years_exact.csv", float_format="%.6g")

    bootstrap_df = evaluation.bootstrap_summary_df(
        pd.DataFrame({**exact_book_dict, **legacy_book_dict, "S&P 500 TR": benchmark_exact_dict["S&P 500 TR"], "60/40": benchmark_exact_dict["60/40"]}),
        benchmark_exact_dict["T-bills"], "60/40",
        replication_count_int=int(spec_dict["tests"]["bootstrap"]["replications_int"]),
        mean_block_length_float=float(spec_dict["tests"]["bootstrap"]["mean_block_days_float"]),
    )
    bootstrap_df.to_csv(BOOK_DIR_PATH / "bootstrap_exact.csv", float_format="%.6g")

    # Perturbation: random engine / member / stabilizer splits, s re-solved to the same target.
    rng = np.random.default_rng(int(spec_dict["tests"]["perturbation"]["seed_int"]))
    concentration_float = float(spec_dict["tests"]["perturbation"]["dirichlet_concentration_float"])
    perturbation_row_list = []
    for product_dict in product_list:
        line_dict = spec_dict["lines"][product_dict["line_str"]]
        engine_list = list(line_dict["return_engines"])
        for draw_int in range(int(spec_dict["tests"]["perturbation"]["draws_int"])):
            engine_share_dict = dirichlet_around(rng, {e: 1.0 / len(engine_list) for e in engine_list}, concentration_float)
            member_split_dict = {e: float(dirichlet_around(rng, {"p": 0.5, "s": 0.5}, concentration_float)["p"]) for e in engine_list}
            stabilizer_dict = dirichlet_around(rng, dict(line_dict["stabilizer_capital"]), concentration_float)
            _, draw_ser = product_weight_ser(spec_dict, product_dict, covariance_exact_df, engine_share_override_dict=engine_share_dict,
                                             member_split_override_dict=member_split_dict, stabilizer_override_dict=stabilizer_dict)
            # The draws themselves are unchanged (same random stream for every product); an amended product drops its sleeves after.
            draw_ser = drop_amended_ser(draw_ser, amendment_by_product_dict.get(product_dict["product_id_str"]))
            draw_book_ser, _ = common.book_return_ser(exact_return_df, (draw_ser / draw_ser.sum()).to_dict(), "annual")
            draw_metric = full_metric_dict(draw_book_ser, benchmark_return_df, exact_base_ts)
            perturbation_row_list.append({"product_id_str": product_dict["product_id_str"], "draw_int": draw_int,
                                          "cagr_float": draw_metric["cagr_float"], "volatility_float": draw_metric["volatility_float"],
                                          "sharpe_rf0_float": draw_metric["sharpe_rf0_float"], "max_drawdown_float": draw_metric["max_drawdown_float"],
                                          "es95_21d_float": draw_metric["es95_21d_float"]})
    perturbation_df = pd.DataFrame(perturbation_row_list)
    perturbation_df.to_csv(BOOK_DIR_PATH / "perturbation_exact.csv", index=False, float_format="%.6g")

    # Estimation split: s re-solved on each half's covariance only, evaluated on the full window.
    split_row_list = []
    half_int = len(exact_index) // 2
    for half_str, half_index in (("first_half", exact_index[:half_int]), ("second_half", exact_index[half_int:])):
        half_covariance_df = weekly_covariance_df(sleeve_return_df.loc[half_index, sorted(exact_alias_set)])
        for product_dict in product_list:
            half_share_float, half_raw_ser = product_weight_ser(spec_dict, product_dict, half_covariance_df)
            half_raw_ser = drop_amended_ser(half_raw_ser, amendment_by_product_dict.get(product_dict["product_id_str"]))
            half_weight_ser = construction.round_weight_ser(half_raw_ser, rounding_step_float)
            half_book_ser, _ = common.book_return_ser(exact_return_df, half_weight_ser.to_dict(), "annual")
            half_metric = full_metric_dict(half_book_ser, benchmark_return_df, exact_base_ts)
            for alias_str, w in half_weight_ser.items():
                split_row_list.append({"estimation_str": half_str, "product_id_str": product_dict["product_id_str"], "alias_str": alias_str,
                                       "weight_float": w, "return_share_float": half_share_float, "book_cagr_float": half_metric["cagr_float"],
                                       "book_volatility_float": half_metric["volatility_float"], "book_sharpe_float": half_metric["sharpe_rf0_float"],
                                       "book_max_drawdown_float": half_metric["max_drawdown_float"]})
    pd.DataFrame(split_row_list).to_csv(BOOK_DIR_PATH / "estimation_split.csv", index=False, float_format="%.6g")

    # Alternative constructions on the same windows.
    alternative_row_list = []
    for product_dict in product_list:
        product_id_str = product_dict["product_id_str"]
        result_dict = product_result_dict[product_id_str]
        alias_list = list(result_dict["weight_dict"])
        vol_arr = np.sqrt(np.diag(covariance_exact_df.loc[alias_list, alias_list]))
        inverse_vol_ser = pd.Series(1.0 / vol_arr, index=alias_list)
        alternative_dict = {
            "equal_risk_engines": drop_amended_ser(
                templates.equal_risk_return_bucket_ser(spec_dict, product_dict["line_str"], covariance_exact_df, result_dict["return_share_float"]),
                amendment_by_product_dict.get(product_id_str)),
            "inverse_volatility": inverse_vol_ser / inverse_vol_ser.sum(),
            "equal_capital": pd.Series(1.0 / len(alias_list), index=alias_list),
            "without_flagged": product_weight_ser(spec_dict, product_dict, covariance_exact_df,
                                                  disabled_secondary_set=set(spec_dict["tests"]["flagged_secondaries_disabled"]))[1],
        }
        for alternative_str, alt_ser in alternative_dict.items():
            alt_ser = alt_ser / alt_ser.sum()
            alt_book_ser, _ = common.book_return_ser(exact_return_df, alt_ser.to_dict(), "annual")
            alternative_row_list.append({"product_id_str": product_id_str, "alternative_str": alternative_str,
                                         "weights_str": json.dumps({k: round(float(v), 3) for k, v in alt_ser.sort_values(ascending=False).items()}),
                                         **full_metric_dict(alt_book_ser, benchmark_return_df, exact_base_ts)})
    # Macro basket: CORE5 anchor shared with Trinity (both BIL-reserve multi-asset sleeves).
    basket_alias_set = exact_alias_set | {"trinity"}
    basket_start_ts = max(exact_start_ts, window_start_ts(sleeve_return_df, ["trinity"]))
    basket_index = sleeve_return_df.loc[basket_start_ts:].index
    basket_covariance_df = weekly_covariance_df(sleeve_return_df.loc[basket_index, sorted(basket_alias_set)])
    basket_spec_dict = json.loads(json.dumps(spec_dict))
    basket_spec_dict["lines"]["main"]["stabilizer_capital"] = dict(spec_dict["tests"]["macro_basket_capital"])
    basket_base_ts = previous_session_ts(sleeve_return_df.index, basket_start_ts)
    for product_dict in [p for p in product_list if p["line_str"] == "main"]:
        _, basket_raw_ser = product_weight_ser(basket_spec_dict, product_dict, basket_covariance_df)
        basket_ser = construction.round_weight_ser(basket_raw_ser, rounding_step_float)
        basket_book_ser, _ = common.book_return_ser(sleeve_return_df.loc[basket_index], basket_ser.to_dict(), "annual")
        alternative_row_list.append({"product_id_str": product_dict["product_id_str"], "alternative_str": "macro_basket",
                                     "weights_str": json.dumps({k: round(float(v), 3) for k, v in basket_ser.sort_values(ascending=False).items()}),
                                     **full_metric_dict(basket_book_ser, benchmark_return_df, basket_base_ts)})
    pd.DataFrame(alternative_row_list).to_csv(BOOK_DIR_PATH / "alternatives_exact.csv", index=False, float_format="%.6g")

    # Hedge overlays: fixed capital slice, funded pro rata, reset annually.
    overlay_row_list = []
    for overlay_dict in spec_dict["tests"]["hedge_overlays"]:
        overlay_alias_str = overlay_dict["alias_str"]
        overlay_weight_float = float(overlay_dict["weight_float"])
        overlay_start_ts = max(exact_start_ts, sleeve_return_df[overlay_alias_str].first_valid_index())
        overlay_index = sleeve_return_df.loc[overlay_start_ts:].index
        overlay_base_ts = previous_session_ts(sleeve_return_df.index, overlay_start_ts)
        for product_id_str in overlay_dict["product_id_list"]:
            base_weight_dict = product_result_dict[product_id_str]["weight_dict"]
            with_dict = {k: v * (1.0 - overlay_weight_float) for k, v in base_weight_dict.items()}
            with_dict[overlay_alias_str] = with_dict.get(overlay_alias_str, 0.0) + overlay_weight_float
            for variant_str, variant_weight_dict in (("without", base_weight_dict), ("with", with_dict)):
                variant_ser, _ = common.book_return_ser(sleeve_return_df.loc[overlay_index], variant_weight_dict, "annual")
                crisis_dict = {f"crisis::{e['label_str']}": common.window_return_float(variant_ser, e["peak_date_str"], e["trough_date_str"])
                               for e in episode_list if pd.Timestamp(e["peak_date_str"]) >= overlay_index[0]}
                overlay_row_list.append({"product_id_str": product_id_str, "overlay_alias_str": overlay_alias_str, "variant_str": variant_str,
                                         **full_metric_dict(variant_ser, benchmark_return_df, overlay_base_ts), **crisis_dict})
    pd.DataFrame(overlay_row_list).to_csv(BOOK_DIR_PATH / "hedge_overlays_exact.csv", index=False, float_format="%.6g")

    # Cost / financing stress and the cash-interest upside, sleeve by sleeve.
    path_by_alias_dict = common.load_sleeve_path_dict()
    rate_ser = evaluation.lagged_tbill_annual_rate_ser(sleeve_return_df.index)
    stress_spec_dict = spec_dict["tests"]["stress"]
    stressed_return_df = sleeve_return_df.copy()
    upside_return_df = sleeve_return_df.copy()
    drag_row_list = []
    for alias_str in sorted(exact_alias_set | set(spec_dict["legacy_books"]["ladder_1_defensive"]) | {"taa_btal_lin_qqq"}):
        path_df = path_by_alias_dict[alias_str].loc[:end_ts]
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        slippage_ser = evaluation.extra_slippage_cost_ser(transaction_df, path_df["total_value_float"], float(stress_spec_dict["extra_slippage_per_side_float"])).reindex(sleeve_return_df.index).fillna(0.0)
        financing_ser = evaluation.financing_cost_ser(path_df, rate_ser, float(stress_spec_dict["financing_spread_over_tbill_float"])).reindex(sleeve_return_df.index).fillna(0.0)
        if alias_str in stress_spec_dict["cash_already_accrues_alias_list"]:
            uplift_ser = pd.Series(0.0, index=sleeve_return_df.index)
        else:
            uplift_ser = evaluation.cash_interest_uplift_ser(path_df, rate_ser, float(stress_spec_dict["cash_rate_haircut_float"])).reindex(sleeve_return_df.index).fillna(0.0)
        live_mask = sleeve_return_df[alias_str].notna()
        stressed_return_df.loc[live_mask, alias_str] = sleeve_return_df.loc[live_mask, alias_str] - slippage_ser[live_mask] - financing_ser[live_mask]
        upside_return_df.loc[live_mask, alias_str] = sleeve_return_df.loc[live_mask, alias_str] + uplift_ser[live_mask]
        exact_mask = live_mask & (sleeve_return_df.index >= exact_start_ts)
        year_count_float = exact_mask.sum() / 252.0
        drag_row_list.append({"alias_str": alias_str,
                              "extra_slippage_drag_per_year_float": float(slippage_ser[exact_mask].sum() / year_count_float),
                              "financing_drag_per_year_float": float(financing_ser[exact_mask].sum() / year_count_float),
                              "cash_uplift_per_year_float": float(uplift_ser[exact_mask].sum() / year_count_float)})
    pd.DataFrame(drag_row_list).to_csv(BOOK_DIR_PATH / "sleeve_stress_drags_exact.csv", index=False, float_format="%.6g")
    stress_row_list = []
    for product_id_str, result_dict in product_result_dict.items():
        for case_str, case_return_df in (("stress", stressed_return_df), ("cash_upside", upside_return_df)):
            case_ser, _ = common.book_return_ser(case_return_df.loc[exact_index], result_dict["weight_dict"], "annual")
            stress_row_list.append({"product_id_str": product_id_str, "case_str": case_str, **full_metric_dict(case_ser, benchmark_return_df, exact_base_ts)})
    pd.DataFrame(stress_row_list).to_csv(BOOK_DIR_PATH / "stress_cases_exact.csv", index=False, float_format="%.6g")

    # Gates.
    headline_by_key_dict = {(r["window_str"], r["series_str"]): r for r in metric_row_list}
    gates_spec_dict = spec_dict["gates"]
    gate_row_list = []
    for product_dict in product_list:
        product_id_str = product_dict["product_id_str"]
        exact_metric = headline_by_key_dict[("exact_annual", product_id_str)]
        long_metric = headline_by_key_dict[("long_annual", product_id_str)]
        dd_budget_float = float(product_dict["max_drawdown_budget_float"])
        boot_row = bootstrap_df.loc[product_id_str]
        perturb_df = perturbation_df[perturbation_df["product_id_str"] == product_id_str]
        own_sub_df = subperiod_frame_df[subperiod_frame_df["series_str"] == product_id_str]
        band_float = float(gates_spec_dict["plateau_sharpe_band_float"])
        within_budget_share_float = float((perturb_df["max_drawdown_float"] >= -dd_budget_float).mean())
        within_band_share_float = float((abs(perturb_df["sharpe_rf0_float"] - exact_metric["sharpe_rf0_float"]) <= band_float).mean())
        gate_row_list.append({
            "product_id_str": product_id_str,
            "G1_drawdown_within_budget_exact_bool": bool(exact_metric["max_drawdown_float"] >= -dd_budget_float),
            "G1_drawdown_within_budget_long_bool": bool(long_metric["max_drawdown_float"] >= -dd_budget_float),
            "G2_beats_tbills_bool": bool(exact_metric["cagr_float"] > exact_metric["tbill_cagr_float"]
                                         and boot_row["prob_excess_cagr_positive_float"] >= float(gates_spec_dict["min_prob_excess_cagr_positive_float"])),
            "G5_plateau_bool": bool(within_budget_share_float >= float(gates_spec_dict["plateau_min_share_within_budget_float"])
                                    and within_band_share_float >= float(gates_spec_dict["plateau_min_share_within_sharpe_band_float"])),
            "G6_positive_excess_every_third_bool": bool((own_sub_df["excess_cagr_float"] > 0).all()),
            "perturbation_share_within_dd_budget_float": within_budget_share_float,
            "perturbation_share_within_sharpe_band_float": within_band_share_float,
            "base_sharpe_percentile_in_perturbation_float": float((perturb_df["sharpe_rf0_float"] < exact_metric["sharpe_rf0_float"]).mean()),
        })
    gate_df = pd.DataFrame(gate_row_list).set_index("product_id_str")
    gate_df.to_csv(BOOK_DIR_PATH / "gates_product.csv", float_format="%.4f")
    ladder_row_list = []
    for line_list in gates_spec_dict["ladder_order_by_line"]:
        for lower_str, upper_str in zip(line_list[:-1], line_list[1:]):
            lower_metric = headline_by_key_dict[("exact_annual", lower_str)]
            upper_metric = headline_by_key_dict[("exact_annual", upper_str)]
            corr_float = float(product_corr_df.loc[lower_str, upper_str])
            ladder_row_list.append({
                "lower_str": lower_str, "upper_str": upper_str,
                "G3_monotone_bool": bool(upper_metric["cagr_float"] > lower_metric["cagr_float"]
                                         and upper_metric["volatility_float"] > lower_metric["volatility_float"]
                                         and upper_metric["max_drawdown_float"] < lower_metric["max_drawdown_float"]
                                         and upper_metric["beta_spx_float"] > lower_metric["beta_spx_float"]),
                "daily_corr_float": corr_float,
                "G4_distinct_bool": bool(corr_float < float(gates_spec_dict["max_adjacent_daily_corr_float"])),
            })
    # Cross-line pairs (main vs low-touch at the same rung) are reported, not gated:
    # the low-touch line differs by trading burden, which is its reason to exist.
    for pair_dict in gates_spec_dict.get("cross_line_pairs", []):
        ladder_row_list.append({"lower_str": pair_dict["main_str"], "upper_str": pair_dict["monthly_str"], "G3_monotone_bool": None,
                                "daily_corr_float": float(product_corr_df.loc[pair_dict["main_str"], pair_dict["monthly_str"]]), "G4_distinct_bool": None})
    pd.DataFrame(ladder_row_list).to_csv(BOOK_DIR_PATH / "gates_ladder.csv", index=False, float_format="%.4f")

    pd.DataFrame({**{k: common.nav_from_return_ser(v) for k, v in exact_book_dict.items()},
                  **{f"drift::{k}": common.nav_from_return_ser(v) for k, v in drift_book_dict.items()},
                  **{f"legacy::{k}": common.nav_from_return_ser(v) for k, v in legacy_book_dict.items()},
                  **{k: common.nav_from_return_ser(v) for k, v in benchmark_exact_dict.items()}}).to_csv(
        BOOK_DIR_PATH / "nav_exact.csv.gz", float_format="%.8g", compression="gzip")
    pd.DataFrame({**{k: common.nav_from_return_ser(v) for k, v in long_book_dict.items()},
                  **{k: common.nav_from_return_ser(v) for k, v in benchmark_long_dict.items()}}).to_csv(
        BOOK_DIR_PATH / "nav_long.csv.gz", float_format="%.8g", compression="gzip")

    run_summary_dict = {
        "exact_window": [exact_start_ts.date().isoformat(), end_ts.date().isoformat(), int(len(exact_index))],
        "long_window": [long_start_ts.date().isoformat(), end_ts.date().isoformat(), int(len(long_index))],
        "products": product_result_dict,
        "spec_sha256_str": SPEC_HASH_PATH.read_text(encoding="utf-8").split()[0],
        "amendments": list(amendment_by_product_dict.values()),
    }
    (BOOK_DIR_PATH / "run_summary.json").write_text(json.dumps(run_summary_dict, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 260)
    print(json.dumps({k: {"s": v["return_share_float"], "vol": round(v["ex_ante_volatility_float"], 4), "w": v["weight_dict"]}
                      for k, v in product_result_dict.items()}, indent=1, default=float))
    show_col_list = ["window_str", "series_str", "cagr_float", "tbill_cagr_float", "volatility_float", "sharpe_rf0_float", "sharpe_excess_float",
                     "max_drawdown_float", "beta_spx_float", "worst_year_float", "worst_12m_float"]
    print(metric_df[show_col_list].round(3).to_string())
    print(gate_df.round(3).to_string())
    print(pd.DataFrame(ladder_row_list).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
