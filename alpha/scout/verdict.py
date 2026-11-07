"""Quant verdict (A27): one page per re-audited strategy that answers the questions an allocator's quant asks, each as
one status line, from results Scout already saved (no re-run).

    python -m alpha.scout verdict [--pod DV2]      -> results/scout/verdict/<pod>.json, verdict.md, index.json

Questions, sources and the fixed thresholds (set before any verdict was read):

1. Is the mechanism real?            literature status of the family (FAMILY_EVIDENCE_DICT) + the S3 edge study.
   green: documented and S3 passes; amber: one of the two weak; red: S3 hard-fails.
2. Does the rule beat luck in its own grid?   S5 MCPT p of the whole search. green <= 0.05, amber <= 0.10.
3. Was choosing it a lucky pick?    S5 correlation-aware null with the family's prior studies (A25/A26), counted up to
   the date the strategy was graded (its bundle's date), each study's trials correlated at 0.3. A floor: pre-Scout
   variants are not registered. green <= 0.05, amber <= 0.20. All studies to date are shown, not judged.
4. Is it still alive?               Sharpe over T-bills of the live configuration (worst rebalance offset) by era:
   to 2015, 2016-2022, 2023 on. 2023 on was seen at the re-audit: a recent check, not a holdout. green: 2023+ > 0 and
   >= half the Sharpe to 2022; amber: 2023+ > 0; grey (n/a): no data after 2022.
5. Does it add beyond known factors?    net alpha t against QQQ + an ETF mix + trend (S6, the house pods excluded).
   green t >= 2, amber t >= 1.5.
6. Does it add to the book?         S6 T-bill slot: probability the book is better with it than with T-bills.
   green >= 0.80, amber >= 0.60.
7. Costs and capacity               net Sharpe at twice the costs + 10 bp (S4) against the net Sharpe.
   green: stressed >= half of net; amber: stressed > 0. Capacity (1% of ADV, last 3 years) shown.
8. How far is the proof?            forward record (shadow log or live pod health) against the minimum track record.
   green: forward months >= MinTRL and status GREEN; red: RED; grey: no MinTRL; amber otherwise (too early).

Research only: reads saved bundles, the ledger, the shadow logs and the pod health report; writes only under
results/scout/verdict.
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger

REAUDITION_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition"
VERDICT_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "verdict"
SHADOW_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "shadow"
POD_HEALTH_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "pod_health"
# The re-audition searches were frozen into the ledger on this date (P5-P7b); the family's studies up to it are the
# search the strategy was chosen from. Pre-Scout studies are not registered (A22), so this count is a floor.
CHRONO_DATE_STR = "2026-10-02"
PRIOR_RHO_FLOAT = 0.3
DRAW_COUNT_INT = 20000
ERA_TUPLE = (("2004-2015", None, "2015-12-31"), ("2016-2022", "2016-01-01", "2022-12-31"), ("2023 on", "2023-01-01", None))
# Strategy folder -> its S8 shadow registration or its live pod label in the pod health report.
SHADOW_BY_POD_DICT = {"EOM": "eom_reaudition_20261002", "DV2": "dv2_reaudition_20261002",
                      "DV2_Nasdaq_100": "dv2_ndx_reaudition_20261002", "DV2_industry_ETF": "dv2_industry_etf_reaudition_20261002"}
LIVE_POD_BY_POD_DICT = {"TAA_3x": "TAA 3x", "NDX_VXN": "NDX VXN"}

# Outside evidence per mechanism family: status word + the references it rests on (A27). ESTABLISHED = published and
# replicated out of sample, including after publication; PRACTITIONER = known in practice, thin academic evidence.
FAMILY_EVIDENCE_DICT: dict[str, tuple[str, str]] = {
    "us_equity_short_term_reversal": ("ESTABLISHED", "Jegadeesh (1990), Lehmann (1990); liquidity provision, Nagel (2012); "
                                      "weaker after publication and in large caps, stronger when volatility is high."),
    "equity_cross_sectional_momentum": ("ESTABLISHED", "Jegadeesh and Titman (1993, 2001 out of sample); 40+ markets, Asness, "
                                        "Moskowitz and Pedersen (2013); crash risk, Daniel and Moskowitz (2016)."),
    "tactical_asset_allocation": ("ESTABLISHED", "Time-series momentum, Moskowitz, Ooi and Pedersen (2012); Faber (2007); "
                                  "cross-asset momentum, Asness, Moskowitz and Pedersen (2013)."),
    "time_series_trend_and_breakout": ("ESTABLISHED", "A century of trend following, Hurst, Ooi and Pedersen (2017)."),
    "calendar_and_flow": ("ESTABLISHED", "Turn of the month, Ariel (1987), Lakonishok and Smidt (1988); month-end "
                          "rebalancing flows, Etula, Rinne, Suominen and Vaittinen (2020)."),
    "volatility_premium_and_hedges": ("ESTABLISHED", "Variance risk premium, Carr and Wu (2009); Bollerslev, Tauchen and Zhou (2009)."),
    "etf_short_term_reversal": ("PRACTITIONER", "Index and ETF oversold rules (IBS, RSI(2), Connors); little academic work."),
    "macro_regime_allocation": ("PRACTITIONER", "Regime allocation by inflation and growth; mixed evidence."),
    "optimized_low_risk_allocation": ("ESTABLISHED", "Low-volatility anomaly, Frazzini and Pedersen (2014); minimum variance."),
    "index_and_listing_events": ("ESTABLISHED", "Index inclusion effects, Shleifer (1986); largely arbitraged since."),
    "dividend_seeking_demand": ("ESTABLISHED", "Dividend month premium, Hartzmark and Solomon (2013)."),
}

_STATUS_RANK_DICT = {"green": 0, "amber": 1, "red": 2}


def _row(key_str: str, question_str: str, status_str: str, headline_str: str, detail_str: str = "") -> dict:
    return {"key_str": key_str, "question_str": question_str, "status_str": status_str, "headline_str": headline_str,
            "detail_str": detail_str}


def graded_date_str_for(pod_str: str) -> str:
    """The date the strategy was graded: its saved bundle was written then, so the family's search up to that date is
    the one it was judged against (review: one fixed date would drop later families' own studies)."""
    path = REAUDITION_ROOT_PATH / pod_str / "bundle.pkl"
    return date.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else CHRONO_DATE_STR


_TBILL_CACHE_DICT: dict = {}


def tbill_return_ser(date_index: pd.DatetimeIndex) -> pd.Series:
    """Daily T-bill return (DTB3, the rate known the session before), the series the re-audition uses."""
    from alpha.scout.metrics import tbill_daily_ser
    from alpha.scout.specs.taa_3x import default_dtb3_csv_path

    if "dtb3" not in _TBILL_CACHE_DICT:
        frame = pd.read_csv(default_dtb3_csv_path())
        _TBILL_CACHE_DICT["dtb3"] = pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(),
                                              index=pd.to_datetime(frame.iloc[:, 0])).dropna().sort_index()
    return tbill_daily_ser(_TBILL_CACHE_DICT["dtb3"], pd.DatetimeIndex(date_index)).fillna(0.0)


def _p_str(p_float: float) -> str:
    """A p-value for a status line: below the resolution of 20,000 draws it reads <0.001, never 0.000."""
    return "<0.001" if p_float < 0.001 else f"{p_float:.3f}"


def _sharpe(return_ser: pd.Series) -> float:
    return_ser = return_ser.dropna()
    if len(return_ser) < 60 or not return_ser.std() > 0:
        return float("nan")
    return float(return_ser.mean() / return_ser.std(ddof=1) * np.sqrt(252.0))


def _s3_status(bundle: dict) -> tuple[str, str]:
    result_dict = (bundle.get("s3") or {}).get("result") or {}
    check_list = result_dict.get("check_list") or []
    verdict_str = str(result_dict.get("verdict_str") or "")
    word_list = [str(c[1]) for c in check_list]
    if verdict_str.startswith("REJECT") or any(w.startswith("FAIL") for w in word_list):
        return "red", verdict_str or "S3 fail"
    if verdict_str.startswith("WATCHLIST") or any(w.startswith("WARN") for w in word_list):
        return "amber", verdict_str or "S3 warn"
    return "green", verdict_str or "S3 pass"


def _forward_dict(pod_str: str, min_trl_float: float | None) -> dict:
    shadow_str = SHADOW_BY_POD_DICT.get(pod_str)
    if shadow_str:
        log_path = SHADOW_ROOT_PATH / shadow_str / "forward_log.csv"
        session_int = 0
        if log_path.exists():
            try:
                session_int = int(pd.read_csv(log_path)["return_float"].notna().sum())
            except (OSError, ValueError, KeyError):
                session_int = 0
        report_list = []
        report_dir_list = sorted((SHADOW_ROOT_PATH / "reports").glob("*/shadow.json")) if (SHADOW_ROOT_PATH / "reports").exists() else []
        if report_dir_list:
            try:
                report_list = json.loads(report_dir_list[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report_list = []
        status_str = next((r.get("status_str") for r in report_list if r.get("pod_id_str") == shadow_str), "n/a")
        return {"source_str": f"shadow {shadow_str}", "months_float": session_int / 21.0, "status_str": status_str}
    live_str = LIVE_POD_BY_POD_DICT.get(pod_str)
    if live_str:
        report_path_list = sorted(POD_HEALTH_ROOT_PATH.glob("*/pod_health.json")) if POD_HEALTH_ROOT_PATH.exists() else []
        for report_path in reversed(report_path_list):
            try:
                report_list = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for report_dict in report_list if isinstance(report_list, list) else []:
                if report_dict.get("label_str") == live_str:
                    sessions_int = int((report_dict.get("live") or {}).get("session_count_int") or 0)
                    return {"source_str": f"live pod {live_str}", "months_float": sessions_int / 21.0,
                            "status_str": report_dict.get("status_str", "n/a")}
            break
    return {"source_str": "none", "months_float": 0.0, "status_str": "n/a"}


def lucky_pick_p_dict(bundle: dict, ledger: Ledger, as_of_date_str: str | None, rho_float: float = PRIOR_RHO_FLOAT) -> dict:
    """Question 3: the S5 correlation-aware null with the family's prior studies drawn per study (A25, A26)."""
    from alpha.scout.reaudit import SEAL_END_STR
    from alpha.scout.stations.s5_overfit import plateau_choice_index_mat
    from alpha.scout.trials import family_trial_summary
    from alpha.stats.psr_dsr import null_selected_sharpe_draws, null_selected_sharpe_p_value

    family_dict = bundle["family"]
    shape_tuple = tuple(len(v) for _, v in sorted(family_dict["grid"].items()))
    grid_df = bundle["s4"].grid_df.loc[:SEAL_END_STR]
    filled_df = grid_df.loc[grid_df.apply(lambda s: s.first_valid_index()).max():].fillna(0.0)
    sharpe_ser = filled_df.mean() / filled_df.std(ddof=1)
    correlation_mat = np.nan_to_num(filled_df.corr().to_numpy(), nan=0.0)
    np.fill_diagonal(correlation_mat, 1.0)
    summary = family_trial_summary(ledger, family_dict["family_id_str"], as_of_date_str=as_of_date_str)
    draw_vec = null_selected_sharpe_draws(
        correlation_mat, len(filled_df), select_index_fn=lambda m: plateau_choice_index_mat(m, shape_tuple),
        prior_group_count_list=list(summary.prior_group_count_tuple), prior_group_correlation_float=rho_float,
        draw_count_int=DRAW_COUNT_INT, random_seed_int=0)
    live_label_str = bundle["s5"].dsr_dict["live"]["label_str"]
    return {"p_float": null_selected_sharpe_p_value(float(sharpe_ser[live_label_str]), draw_vec),
            "studies_int": len(summary.prior_group_count_tuple), "trials_int": summary.prior_trial_count_int}


def build_verdict(pod_str: str, ledger: Ledger | None = None) -> dict:
    ledger = ledger or Ledger()
    bundle = pickle.load(open(REAUDITION_ROOT_PATH / pod_str / "bundle.pkl", "rb"))
    family_id_str = bundle["family"]["family_id_str"]
    s4, s5, s6 = bundle["s4"], bundle["s5"], bundle["s6"]
    row_list = []

    # 1. mechanism
    evidence_word_str, evidence_ref_str = FAMILY_EVIDENCE_DICT.get(family_id_str, ("NONE", "No outside evidence recorded."))
    s3_status_str, s3_text_str = _s3_status(bundle)
    if s3_status_str == "red":
        status_str = "red"
    elif evidence_word_str == "ESTABLISHED" and s3_status_str == "green":
        status_str = "green"
    else:
        status_str = "amber"
    row_list.append(_row("mechanism", "Is the mechanism real?", status_str,
                         f"{evidence_word_str.capitalize()} in the literature; our edge study: {s3_text_str}", evidence_ref_str))

    # 2. own grid
    mcpt_p_float = max(float(c.p_value_float) for c in s5.mcpt_list) if s5.mcpt_list and hasattr(s5.mcpt_list[0], "p_value_float") else None
    if mcpt_p_float is None:
        detail_str = next((c[2] for c in s5.check_list if c[0].startswith("MCPT")), "")
        mcpt_p_float = float(detail_str.split(" p ")[-1].split()[0]) if " p " in detail_str else float("nan")
    status_str = "green" if mcpt_p_float <= 0.05 else ("amber" if mcpt_p_float <= 0.10 else "red")
    plateau_dict = s4.plateau_dict
    row_list.append(_row("own_grid", "Does the rule beat luck in its own grid?", status_str,
                         f"MCPT p {_p_str(mcpt_p_float)}",
                         f"Plateau ratio {plateau_dict.get('plateau_ratio_float', float('nan')):.2f}; "
                         f"{len(s4.sharpe_ser)} configurations."))

    # 3. lucky pick
    graded_date_str = graded_date_str_for(pod_str)
    chrono_dict = lucky_pick_p_dict(bundle, ledger, graded_date_str)
    all_dict = lucky_pick_p_dict(bundle, ledger, None)
    p_float = chrono_dict["p_float"]
    status_str = "green" if p_float <= 0.05 else ("amber" if p_float <= 0.20 else "red")
    row_list.append(_row("lucky_pick", "Was choosing it a lucky pick?", status_str,
                         f"p {_p_str(p_float)} against the recorded search up to {graded_date_str}: "
                         f"{chrono_dict['studies_int']} family studies, {chrono_dict['trials_int']:,} trials",
                         f"A floor: unregistered pre-Scout variants are not counted. Studies correlated at {PRIOR_RHO_FLOAT}. "
                         f"The family-wide question (is the best of everything tried real?), shown, not judged: p "
                         f"{_p_str(all_dict['p_float'])} against {all_dict['studies_int']} studies ({all_dict['trials_int']:,} trials)."))

    # 4. still alive
    raw_ser = s4.luck_dict["live"]["worst_offset_return_ser"].dropna()
    excess_ser = raw_ser - tbill_return_ser(raw_ser.index)  # T-bills earn 4-5% since 2023: raw Sharpe flatters idle sleeves
    first_ts = raw_ser.index.min()
    era_dict = {}
    for label_str, start_str, end_str in ERA_TUPLE:
        era_start_ts = max(first_ts, pd.Timestamp(start_str)) if start_str else first_ts
        if end_str and era_start_ts > pd.Timestamp(end_str):
            continue  # the history starts after this era
        if start_str is None:
            label_str = f"{era_start_ts.year}-2015"
        era_dict[label_str] = _sharpe(excess_ser.loc[era_start_ts:end_str])
    in_sample_float = _sharpe(excess_ser.loc[:"2022-12-31"])
    recent_float = era_dict.get("2023 on", float("nan"))
    if not np.isfinite(recent_float):
        status_str = "grey"  # no data after 2022 in the bundle (e.g. a family whose vault stays closed)
    elif recent_float > 0 and recent_float >= 0.5 * in_sample_float:
        status_str = "green"
    elif recent_float > 0:
        status_str = "amber"
    else:
        status_str = "red"
    row_list.append(_row("alive", "Is it still alive?", status_str,
                         " / ".join(f"{k} {v:.2f}" if np.isfinite(v) else f"{k} n/a" for k, v in era_dict.items()),
                         f"Sharpe over T-bills of the live configuration (worst rebalance offset); to 2022: {in_sample_float:.2f}. "
                         f"2023 on was already seen when the strategy was re-audited: a recent check, not a holdout."))

    # 5. beyond known factors (house pods excluded)
    factor_list = [s for s in s6.spanning_list if "TAA" not in s["model_str"] and "NDX" not in s["model_str"]
                   and "DV2" not in s["model_str"]]
    span_dict = factor_list[-1] if factor_list else (s6.spanning_list[0] if s6.spanning_list else {})
    alpha_t_float = float(span_dict.get("net_alpha_t_float", float("nan")))
    status_str = "green" if alpha_t_float >= 2 else ("amber" if alpha_t_float >= 1.5 else "red")
    house_list = [s for s in s6.spanning_list if s not in factor_list]
    house_str = "; ".join(f"with {s['model_str']}: {s['net_alpha_annual_float']:+.1%}, t {s['net_alpha_t_float']:.2f}" for s in house_list)
    row_list.append(_row("factors", "Does it add beyond known factors?", status_str,
                         f"net alpha {span_dict.get('net_alpha_annual_float', float('nan')):+.1%} a year, t {alpha_t_float:.2f} "
                         f"against {span_dict.get('model_str', 'n/a')}", house_str))

    # 6. book
    slot_dict = s6.slot_dict
    p_better_float = float(slot_dict.get("probability_better_float", float("nan")))
    status_str = "green" if p_better_float >= 0.80 else ("amber" if p_better_float >= 0.60 else "red")
    row_list.append(_row("book", "Does it add to the book?", status_str,
                         f"P(better than T-bills in the slot) {p_better_float:.2f}",
                         f"{slot_dict.get('book_str', '')}: book Sharpe {slot_dict.get('book_sharpe_with_float', float('nan')):.2f} "
                         f"vs {slot_dict.get('book_sharpe_tbills_float', float('nan')):.2f} from {slot_dict.get('start_str', '')}."))

    # 7. costs and capacity
    cost_dict = s4.cost_dict["live"]
    net_float, stressed_float = float(cost_dict["net_sharpe_float"]), float(cost_dict["stressed_sharpe_float"])
    status_str = "green" if stressed_float >= 0.5 * net_float else ("amber" if stressed_float > 0 else "red")
    capacity_dict = s6.capacity_dict or {}
    row_list.append(_row("costs", "Does it survive costs, and how big can it get?", status_str,
                         f"net Sharpe {net_float:.2f}, at twice the costs {stressed_float:.2f}",
                         f"Turnover {cost_dict.get('annual_turnover_float', float('nan')):.0f}x a year; capacity "
                         f"${capacity_dict.get('recent_3y_aum_float', float('nan')) / 1e6:.1f}M at 1% of ADV "
                         f"(binding: {capacity_dict.get('binding_asset_str', 'n/a')})."))

    # 8. forward proof
    min_trl_float = (bundle.get("post_adoption") or {}).get("min_track_record_months_float")
    forward_dict = _forward_dict(pod_str, min_trl_float)
    forward_status_str = str(forward_dict["status_str"]).upper()
    has_min_trl_bool = min_trl_float is not None and np.isfinite(min_trl_float) and min_trl_float > 0
    if forward_status_str == "RED":
        status_str = "red"
    elif not has_min_trl_bool:
        status_str = "grey"
    elif forward_dict["months_float"] >= min_trl_float and forward_status_str in ("GREEN", "OK"):
        status_str = "green"
    else:
        status_str = "amber"
    min_trl_str = f"{min_trl_float:.0f}" if has_min_trl_bool else "n/a"
    row_list.append(_row("forward", "How far is the forward proof?", status_str,
                         f"{forward_dict['months_float']:.1f} of {min_trl_str} months needed "
                         f"({forward_dict['source_str']}, {forward_dict['status_str']})",
                         "Minimum track record to confirm the backtest Sharpe at 95% (MinTRL)."))

    return {"pod_str": pod_str, "family_id_str": family_id_str, "built_str": date.today().isoformat(),
            "live_label_str": s5.dsr_dict["live"]["label_str"], "row_list": row_list,
            "worst_status_str": max((r["status_str"] for r in row_list if r["status_str"] in _STATUS_RANK_DICT),
                                    key=lambda s: _STATUS_RANK_DICT[s], default="grey"),
            "count_dict": {s: sum(r["status_str"] == s for r in row_list) for s in (*_STATUS_RANK_DICT, "grey")}}


def render_markdown(verdict_list: list[dict]) -> str:
    line_list = ["# Scout quant verdicts", "", "One row per question; thresholds in alpha/scout/verdict.py (A27).", ""]
    for verdict_dict in verdict_list:
        line_list += [f"## {verdict_dict['pod_str']} ({verdict_dict['family_id_str']})", "",
                      "| Question | Status | Answer |", "| --- | --- | --- |"]
        line_list += [f"| {r['question_str']} | {r['status_str']} | {r['headline_str']} |" for r in verdict_dict["row_list"]]
        line_list.append("")
    return "\n".join(line_list)


def write_verdicts(pod_list: list[str] | None = None, ledger: Ledger | None = None,
                   output_root_path: Path = VERDICT_ROOT_PATH) -> list[dict]:
    ledger = ledger or Ledger()
    pod_list = pod_list or sorted(p.parent.name for p in REAUDITION_ROOT_PATH.glob("*/bundle.pkl"))
    output_root_path.mkdir(parents=True, exist_ok=True)
    verdict_list = []
    for pod_str in pod_list:
        try:
            verdict_dict = build_verdict(pod_str, ledger)
        except Exception as error:  # one broken bundle must not stop the others
            verdict_dict = {"pod_str": pod_str, "error_str": f"{type(error).__name__}: {error}", "row_list": []}
        (output_root_path / f"{pod_str}.json").write_text(json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
        verdict_list.append(verdict_dict)
    index_list = [{"pod_str": v["pod_str"], "family_id_str": v.get("family_id_str"), "worst_status_str": v.get("worst_status_str"),
                   "count_dict": v.get("count_dict"), "error_str": v.get("error_str")} for v in verdict_list]
    (output_root_path / "index.json").write_text(json.dumps(index_list, indent=2), encoding="utf-8")
    (output_root_path / "verdict.md").write_text(render_markdown([v for v in verdict_list if not v.get("error_str")]), encoding="utf-8")
    return verdict_list
