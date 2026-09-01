"""Build the research-only SPY Adaptive Momentum regime evidence bundle.

Workflow:

1. ``--prepare-data`` freezes a Norgate cache without calculating performance.
2. ``--write-contract`` writes the source map and frozen specification.
3. Validate the contract and adaptive sidecars with the research-skill tools.
4. ``--run`` calculates evidence and writes reports, charts, and a notebook.

No LIVE, broker, scheduler, release, allocation, or registry path is read or
modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor


MODULE_ALPHA_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(MODULE_ALPHA_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_ALPHA_ROOT_PATH))

from strategies.momentum.strategy_mo_spy_adaptive_momentum_regime import (
    DEFAULT_CONFIG,
    SpyAdaptiveMomentumRegimeConfig,
    compute_spy_adaptive_momentum_signal_df,
    get_spy_adaptive_momentum_regime_data,
)


ALPHA_ROOT_PATH = MODULE_ALPHA_ROOT_PATH
PAKAL_ROOT_PATH = ALPHA_ROOT_PATH.parent / "pakal"
STUDY_ID_STR = "spy_adaptive_momentum_regime_study"
STUDY_DIR_PATH = PAKAL_ROOT_PATH / "pakal-research" / "reports" / STUDY_ID_STR
DATA_DIR_PATH = STUDY_DIR_PATH / "data"
TABLE_DIR_PATH = STUDY_DIR_PATH / "tables"
CHART_DIR_PATH = STUDY_DIR_PATH / "charts"
DATA_CACHE_PATH = DATA_DIR_PATH / "spy_adaptive_momentum_norgate.csv"
DATA_METADATA_PATH = DATA_DIR_PATH / "data_snapshot.json"
SPEC_PATH = STUDY_DIR_PATH / "research_spec_frozen.json"
NOTEBOOK_PATH = PAKAL_ROOT_PATH / "pakal-research" / f"{STUDY_ID_STR}.ipynb"

SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
SOURCE_PART_1_PATH = SOURCE_DIR_PATH / "adaptive_mom_vardi_pt1.pdf"
SOURCE_PART_2_PATH = SOURCE_DIR_PATH / "adaptive_mom_vardi_pt2.pdf"
PASTED_NOTE_PATH = SOURCE_DIR_PATH / "user_pasted_commentary.txt"

EVALUATION_START_STR = "1995-01-03"
SOURCE_SAMPLE_END_STR = "2020-12-31"
VALIDATION_START_STR = "2021-01-07"
VALIDATION_END_STR = "2023-12-29"
CONFIRMATION_START_STR = "2024-01-02"
ANNUALIZATION_FLOAT = 252.0
ACTIVE_MINUTES_USED_INT = 45

COST_ROUND_TRIP_BPS_DICT = {
    "paper_like": 0.0,
    "central_research": 10.0,
    "conservative_survival": 25.0,
}

SOURCE_REPORTED_METRIC_DICT = {
    "CAGR": 0.108,
    "annualized_volatility": 0.135,
    "Sharpe": 0.80,
    "maximum_drawdown": -0.27,
    "state_change_count": 138,
}


@dataclass(frozen=True)
class ResearchVariant:
    variant_key_str: str
    percentile_lookback_int: int = 126
    high_lookback_int: int | None = None
    percentile_method_str: str = "strict"
    fast_lookback_int: int = 50
    slow_lookback_int: int = 200
    percentile_power_float: float = 2.0
    price_filter_lookback_int: int = 10
    provenance_str: str = "source_literal"


SOURCE_VARIANT_OBJ = ResearchVariant(variant_key_str="source_exact")
RESEARCH_VARIANT_LIST = [
    SOURCE_VARIANT_OBJ,
    replace(SOURCE_VARIANT_OBJ, variant_key_str="percentile_63", percentile_lookback_int=63, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="percentile_252", percentile_lookback_int=252, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="fast_25", fast_lookback_int=25, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="fast_75", fast_lookback_int=75, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="slow_150", slow_lookback_int=150, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="slow_250", slow_lookback_int=250, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="power_1", percentile_power_float=1.0, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="power_3", percentile_power_float=3.0, provenance_str="predeclared_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="filter_3", price_filter_lookback_int=3, provenance_str="source_stated_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="filter_15", price_filter_lookback_int=15, provenance_str="source_stated_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="high_252", high_lookback_int=252, provenance_str="source_stated_robustness"),
    replace(SOURCE_VARIANT_OBJ, variant_key_str="weak_ecdf_ties", percentile_method_str="weak", provenance_str="source_ambiguity"),
]


def now_iso_str() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file_str(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for byte_chunk in iter(lambda: file_obj.read(1 << 20), b""):
            hash_obj.update(byte_chunk)
    return hash_obj.hexdigest()


def validate_source_files_and_hashes(
    expected_hash_dict: dict[str, object] | None = None,
) -> dict[str, str]:
    source_path_dict = {
        "part_1": SOURCE_PART_1_PATH,
        "part_2": SOURCE_PART_2_PATH,
        "pasted_note": PASTED_NOTE_PATH,
    }
    missing_source_path_list = [
        source_path
        for source_path in source_path_dict.values()
        if not source_path.is_file()
    ]
    if missing_source_path_list:
        missing_source_str = ", ".join(str(source_path) for source_path in missing_source_path_list)
        raise FileNotFoundError(
            "Research source files are required. Supply --source-part-1-path, "
            "--source-part-2-path, and --pasted-note-path, or place the files "
            f"in the study sources directory. Missing: {missing_source_str}"
        )

    actual_hash_dict = {
        source_id_str: sha256_file_str(source_path)
        for source_id_str, source_path in source_path_dict.items()
    }
    if expected_hash_dict is not None:
        mismatch_source_id_list = [
            source_id_str
            for source_id_str, actual_hash_str in actual_hash_dict.items()
            if str(expected_hash_dict.get(source_id_str, "")) != actual_hash_str
        ]
        if mismatch_source_id_list:
            raise RuntimeError(
                "Research source hashes do not match data_snapshot.json: "
                + ", ".join(mismatch_source_id_list)
            )
    return actual_hash_dict


def write_json(file_path: Path, payload_obj: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_data_cache() -> dict[str, object]:
    source_hash_dict = validate_source_files_and_hashes()
    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    pricing_data_df = get_spy_adaptive_momentum_regime_data(DEFAULT_CONFIG)
    required_column_tuple = (
        (DEFAULT_CONFIG.trade_symbol_str, "Open"),
        (DEFAULT_CONFIG.trade_symbol_str, "Close"),
        (DEFAULT_CONFIG.trade_symbol_str, "Unadjusted Close"),
        (DEFAULT_CONFIG.trade_symbol_str, "Dividend"),
        (DEFAULT_CONFIG.trade_symbol_str, "Volume"),
        (DEFAULT_CONFIG.trade_symbol_str, "Turnover"),
        (DEFAULT_CONFIG.signal_symbol_str, "Open"),
        (DEFAULT_CONFIG.signal_symbol_str, "Close"),
        (DEFAULT_CONFIG.benchmark_symbol_str, "Close"),
    )
    missing_column_list = [
        column_tuple
        for column_tuple in required_column_tuple
        if column_tuple not in pricing_data_df.columns
    ]
    if missing_column_list:
        raise RuntimeError(f"Missing required Norgate columns: {missing_column_list}")

    previous_cache_df = (
        pd.read_csv(DATA_CACHE_PATH, parse_dates=["date"]).set_index("date")
        if DATA_CACHE_PATH.exists()
        else None
    )
    previous_cache_sha256_str = (
        sha256_file_str(DATA_CACHE_PATH) if DATA_CACHE_PATH.exists() else None
    )
    cache_df = pd.DataFrame(
        {
            "spy_execution_open": pricing_data_df[("SPY", "Open")],
            "spy_execution_close": pricing_data_df[("SPY", "Close")],
            "spy_unadjusted_close": pricing_data_df[("SPY", "Unadjusted Close")],
            "spy_dividend": pricing_data_df[("SPY", "Dividend")],
            "spy_volume": pricing_data_df[("SPY", "Volume")],
            "spy_turnover": pricing_data_df[("SPY", "Turnover")],
            "spy_total_return_open": pricing_data_df[("SPY_TR_SIGNAL", "Open")],
            "spy_total_return_close": pricing_data_df[("SPY_TR_SIGNAL", "Close")],
            "spx_total_return_close": pricing_data_df[("$SPX", "Close")],
        },
        index=pricing_data_df.index,
        dtype=float,
    ).sort_index()
    cache_df.index.name = "date"
    common_columns_unchanged_bool = True
    if previous_cache_df is not None:
        common_column_list = [
            column_str
            for column_str in previous_cache_df.columns
            if column_str in cache_df.columns
        ]
        serialized_common_df = cache_df[common_column_list].map(
            lambda value_float: (
                float(f"{float(value_float):.12g}")
                if np.isfinite(float(value_float))
                else float(value_float)
            )
        )
        try:
            pd.testing.assert_frame_equal(
                previous_cache_df[common_column_list],
                serialized_common_df,
                check_exact=True,
                check_dtype=False,
            )
        except AssertionError as error_obj:
            common_columns_unchanged_bool = False
            raise RuntimeError(
                "Existing frozen price columns changed while adding capacity volume."
            ) from error_obj
    cache_df.to_csv(
        DATA_CACHE_PATH,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    metadata_dict = {
        "schema_version": "spy-adaptive-momentum-data-v1",
        "created_at": now_iso_str(),
        "vendor": "Norgate Data",
        "first_observation": cache_df.index.min().date().isoformat(),
        "last_observation": cache_df.index.max().date().isoformat(),
        "row_count": len(cache_df),
        "cache_path": str(DATA_CACHE_PATH),
        "cache_sha256": sha256_file_str(DATA_CACHE_PATH),
        "previous_cache_sha256": previous_cache_sha256_str,
        "common_columns_unchanged": common_columns_unchanged_bool,
        "execution_adjustment": "CAPITALSPECIAL with separate Dividend field",
        "signal_adjustment": "TOTALRETURN",
        "benchmark_data_symbol": "$SPXTR returned under $SPX",
        "norgate_loader_attrs": dict(pricing_data_df.attrs),
        "source_sha256": source_hash_dict,
    }
    write_json(DATA_METADATA_PATH, metadata_dict)
    return metadata_dict


def load_cache_df() -> pd.DataFrame:
    cache_df = pd.read_csv(DATA_CACHE_PATH, parse_dates=["date"])
    cache_df = cache_df.set_index("date").sort_index()
    if cache_df.index.has_duplicates:
        raise RuntimeError("Frozen data cache has duplicate dates.")
    return cache_df


def write_source_rule_map(metadata_dict: dict[str, object]) -> None:
    source_hash_dict = metadata_dict["source_sha256"]
    source_map_str = f"""# SPY Adaptive Momentum Market Regime - source rule map

## Source identity

| Source | Content ID | Complete read | Role |
| --- | --- | --- | --- |
| `adaptive_mom_vardi_pt1.pdf` | `sha256:{source_hash_dict['part_1']}` | yes, all 10 pages visually inspected and OCR checked | Literal SPY rule, formulas, results, and author clarifications in comments |
| `adaptive_mom_vardi_pt2.pdf` | `sha256:{source_hash_dict['part_2']}` | yes, all 3 pages visually inspected and OCR checked | Cross-asset transfer evidence and parameter-optimization caveat |
| `pasted-text.txt` | `sha256:{source_hash_dict['pasted_note']}` | yes | User-supplied commentary only; never treated as source authority |

## Literal rules

| Field | Source statement | Internal interpretation | Status |
| --- | --- | --- | --- |
| Objective | Adapt trend speed to non-stationary market moves | Test a drawdown-controlled binary SPY regime | known |
| Universe | SPY for the first post; DBC, TLT, EFA in Part 2 | SPY only, per user request | known |
| Data and adjustment | Comment code uses `SPY.Adjusted` | Norgate TOTALRETURN for signal; CAPITALSPECIAL plus dividend ledger for engine execution | proxy disclosed |
| Drawdown | Drawdown from all-time high; one-year high also said to work | Primary uses all-time high; 252-session high is one predeclared robustness row | known with alternative |
| Severity rank | Percentile-rank drawdowns over the past six months, then square | Primary uses strict ECDF of `-drawdown` over 126 observed sessions; weak ECDF is an ambiguity row | tie handling missing |
| Adaptive speed | `P*ST + (1-P)*LT`, using EMA alpha | `P=q^2`, `alpha_fast=2/(50+1)`, `alpha_slow=2/(200+1)` | clarified in author comment |
| Adaptive average | Same recursion as an exponential moving average | `AMA_T=alpha_T*Close_T+(1-alpha_T)*AMA_(T-1)` | known |
| Price filter | Ten-day moving average; author says 3 or 15 is similar | Primary SMA10; SMA3 and SMA15 are declared robustness rows | known |
| Trading rule | Long when SMA10 is above AMA; flat otherwise | Binary 100% SPY / 0% cash | known |
| Decision time | Daily signal at close; compare both values at the same close | Decision after final Close_T | known |
| Fill time | Not specified; author says no one-day shift in the displayed comparison | Same-close/close-to-close is diagnostic; primary fills Open_(T+1) | ambiguous and timing-conflicted |
| Cash | Long/flat; cash return not specified | Idle cash earns 0% | missing, conservative proxy |
| Costs | Not reported | 0/10/25 bps round-trip tiers on changed notional | missing |
| Parameters searched | Author says choices are not critical and later says optimization helps, without a grid | Thirteen total predeclared source/one-factor variants; no combination search | incomplete |
| Evaluation | SPY chart/table states 1995-2020 | Source-like 1995-2020 plus source-unseen 2021-2023 validation and 2024+ confirmation | known with current extension |

## Timing boundary

```text
SPY Total Return Close_T -> binary decision after Close_T
                              |
                              | *** CRITICAL *** no final Close_T input
                              | may earn a return before Open_(T+1)
                              v
                         SPY Open_(T+1)
```

## Reproduction gaps and proxies

| Gap or proxy | Consequence | Treatment |
| --- | --- | --- |
| Percentile rank direction and ties are not specified | ATH days and repeated drawdowns can receive different adaptive speeds | Strict severity ECDF is primary; weak ECDF shown separately |
| EMA seed is not specified | Early values can differ | Seed at first valid price; start evaluation after nearly two years of warm-up |
| Exact sample endpoint and data vendor are absent | Local source-period metrics may not match exactly | Classify reproduction with explicit tolerances and Norgate lineage |
| Source fill and cost are absent | Displayed result may include same-close timing conflict | Separate source-like, same-exit intraday, and primary next-open paths |
| Trade-count definition is absent | Entry, exit, and state-change counts may differ | Report all state changes and label the source comparison approximate |

## Methodological lessons separated from literal rules

- Drawdown severity can schedule filter bandwidth without fitting a classifier.
- Faster rebound response is the proposed mechanism, not proof of incremental alpha.
- Part 2 is transfer evidence, not untouched validation and not authority to trade other assets here.
"""
    (STUDY_DIR_PATH / "SOURCE_RULE_MAP.md").write_text(
        source_map_str,
        encoding="utf-8",
    )


def write_frozen_contract() -> None:
    if not DATA_METADATA_PATH.exists():
        raise FileNotFoundError("Prepare the immutable data cache first.")
    metadata_dict = json.loads(DATA_METADATA_PATH.read_text(encoding="utf-8"))
    validate_source_files_and_hashes(dict(metadata_dict["source_sha256"]))
    write_source_rule_map(metadata_dict)
    frozen_at_str = now_iso_str()
    last_observation_str = str(metadata_dict["last_observation"])
    spec_dict = {
        "schema_version": "quant-research-spec-v1",
        "study_id": STUDY_ID_STR,
        "status": "frozen before any full strategy-performance calculation",
        "initial_frozen_at": frozen_at_str,
        "research_only": True,
        "objective": (
            "Determine whether Varadi's literal drawdown-adaptive SPY momentum regime "
            "survives causal next-open execution and costs, improves risk-adjusted "
            "performance versus static trend baselines, and remains stable in "
            "source-unseen validation and confirmation periods."
        ),
        "sources": [
            {
                "source_id": "varadi_adaptive_momentum_part_1",
                "location": str(SOURCE_PART_1_PATH),
                "content_id": f"sha256:{metadata_dict['source_sha256']['part_1']}",
                "role": "Literal SPY methodology, table, and author comments",
                "read_complete": True,
            },
            {
                "source_id": "varadi_adaptive_momentum_part_2",
                "location": str(SOURCE_PART_2_PATH),
                "content_id": f"sha256:{metadata_dict['source_sha256']['part_2']}",
                "role": "Cross-asset extension and parameter caveats",
                "read_complete": True,
            },
            {
                "source_id": "user_pasted_commentary",
                "location": str(PASTED_NOTE_PATH),
                "content_id": f"sha256:{metadata_dict['source_sha256']['pasted_note']}",
                "role": "Commentary only, not source authority",
                "read_complete": True,
            },
        ],
        "data": {
            "vendor": "Norgate Data",
            "period": f"{metadata_dict['first_observation']} through {last_observation_str}",
            "as_of": f"{last_observation_str} final daily observation; cache created {metadata_dict['created_at']}",
            "input_artifacts": [
                {
                    "location": str(DATA_CACHE_PATH),
                    "content_id": f"sha256:{metadata_dict['cache_sha256']}",
                    "as_of": last_observation_str,
                }
            ],
            "universes": ["Fixed SPY ETF; no constituent selection"],
            "benchmarks": [
                "SPY total-return buy-and-hold on the identical open-to-open decision dates",
                "200-session SMA trend",
                "252-session time-series momentum",
                "120-session time-series momentum",
            ],
            "adjustment": (
                "Norgate TOTALRETURN SPY for signal and timing-research returns; "
                "CAPITALSPECIAL SPY plus separate Dividend for engine execution; "
                "$SPXTR returned under $SPX for report benchmark provenance"
            ),
            "membership": "Not applicable to one fixed ETF; no survivorship selection",
        },
        "timing": {
            "decision": "After final SPY Total Return Close_T",
            "entry": "Open_(T+1) for the primary executable path",
            "exit": "Daily target remains until the next target can change at Open_(T+2)",
            "critical_boundary": "No Close_T-derived signal may earn any return before Open_(T+1)",
            "terminal_value": "Drop decisions without the required future open; idle cash earns 0%",
            "diagnostics": [
                "Close_T to Close_(T+1) source-like path",
                "Open_(T+1) to Close_(T+1) same-exit executable attribution",
                "Open_(T+1) to Open_(T+2) primary daily implementation",
            ],
        },
        "signal": {
            "formula": (
                "DD_T=TRClose_T/cummax(TRClose)_T-1; severity_T=-DD_T; "
                "q_T=strict trailing 126-session ECDF severity percentile; Q_T=q_T^2; "
                "alpha_T=Q_T*2/(50+1)+(1-Q_T)*2/(200+1); "
                "AMA_T=alpha_T*TRClose_T+(1-alpha_T)*AMA_(T-1); "
                "filtered_T=SMA10(TRClose)_T; target_T=1[filtered_T>AMA_T]"
            ),
            "thresholds": [
                "126-session severity ECDF",
                "power 2",
                "EMA endpoints 50 and 200",
                "SMA10 price filter",
                "strict greater-than entry",
            ],
        },
        "feature_roles": [
            {
                "feature": "drawdown severity percentile",
                "roles": ["market-regime state", "adaptive filter-speed scheduler"],
            },
            {
                "feature": "SMA10 above adaptive EMA",
                "roles": ["binary SPY exposure signal"],
            },
        ],
        "portfolio": {
            "engine": "Single-asset stateful daily target ledger",
            "maximum_positions": 1,
            "ranking": "Not applicable",
            "sizing": "100% SPY in risk-on, 0% SPY in risk-off",
            "cash": "Idle cash earns 0%; no cash ETF",
            "ensemble": "None",
        },
        "evaluation": {
            "periods": {
                "discovery": f"{EVALUATION_START_STR} to {SOURCE_SAMPLE_END_STR}; source-contaminated",
                "source_discovery": f"{EVALUATION_START_STR} to {SOURCE_SAMPLE_END_STR}; source-contaminated",
                "validation": f"{VALIDATION_START_STR} to {VALIDATION_END_STR}; source-unseen and opened once",
                "confirmation": f"{CONFIRMATION_START_STR} to latest usable decision before {last_observation_str}; locked before run",
                "full": f"{EVALUATION_START_STR} to latest usable decision before {last_observation_str}",
            },
            "metrics": [
                "CAGR",
                "annualized volatility",
                "zero-rate Sharpe",
                "maximum drawdown",
                "Calmar",
                "daily and monthly market correlation",
                "market beta",
                "average exposure",
                "annualized target turnover",
                "state-change count",
                "cost drag",
                "selected-order participation",
            ],
            "inference_unit": "Trading decision date; no pooled cross section",
            "benchmark": "SPY total-return buy-and-hold on exact shared dates plus three static trend rules",
        },
        "costs": {
            "paper_like": {"round_trip_bps": 0.0, "included_components": ["Source omission of costs"]},
            "central_research": {"round_trip_bps": 10.0, "included_components": ["Provisional all-in commission, spread, and ordinary slippage"]},
            "conservative_survival": {"round_trip_bps": 25.0, "included_components": ["Difficult all-in survival hurdle"]},
            "components": [
                "Round-trip bps divided by two and charged per one-way changed target notional",
                "No borrow, financing, FX, or leverage",
                "Opening-auction basis risk and impact excluded from base tiers",
            ],
            "capacity_impact": {
                "separate_from_base": True,
                "formula": "abs(delta_target)*AUM/prior_20_session_mean_SPY_dollar_volume",
                "calibrated": False,
            },
        },
        "search_space": {
            "declared_families": [
                "One literal source configuration",
                "Twelve one-factor source or mechanism robustness configurations",
                "Three static trend baselines",
                "Three timing paths and three fixed cost layers as evaluation dimensions, not selection axes",
            ],
            "axes": {
                "percentile_lookback": [63, 126, 252],
                "fast_lookback_one_factor": [25, 50, 75],
                "slow_lookback_one_factor": [150, 200, 250],
                "percentile_power_one_factor": [1.0, 2.0, 3.0],
                "price_filter_one_factor": [3, 10, 15],
                "reference_high": ["all_time", 252],
                "percentile_ties": ["strict", "weak"],
                "static_baseline": ["SMA200", "MOM252", "MOM120"],
            },
            "combination_rule": "One factor differs from source_exact at a time; no Cartesian product",
            "adaptive_configuration_count": len(RESEARCH_VARIANT_LIST),
            "total_declared_variants": len(RESEARCH_VARIANT_LIST) + 3,
            "multiple_testing": "Report every row; no historical winner replaces source_exact or earns untouched status",
        },
        "promotion_rule": {
            "economic": (
                "At central cost in both validation and confirmation, source_exact must retain at least 70% of SPY CAGR; "
                "conservative full-period CAGR must remain positive"
            ),
            "risk": (
                "At central cost in both validation and confirmation, source_exact maximum-drawdown magnitude must be no more than 80% of SPY"
            ),
            "comparative": (
                "At central cost in both validation and confirmation, source_exact Sharpe must be at least the best of SMA200, MOM252, and MOM120"
            ),
            "statistical": (
                "No standalone p-value promotion is permitted for a single serially dependent market path; "
                "all 13 frozen adaptive rows must be reported and source_exact cannot be replaced by a realized winner"
            ),
            "promotion_limit": (
                "Even a full pass is at most research_candidate; no PAPER/LIVE or allocation authority without forward shadow and empirical open fills"
            ),
        },
        "known_limits": [
            "The source is a blog research note, not a peer-reviewed validation study.",
            "Percentile direction, tie handling, EMA seed, exact sample endpoint, trade count, costs, and fill price are not fully specified.",
            "The source saw 1995-2020; that period is discovery, not confirmation.",
            "Norgate current history is not a frozen archive of what the source downloaded in 2020.",
            "Open auction spread, queue, partial fills, and empirical impact are not measured.",
            "A one-ETF regime filter changes beta and cash time; lower drawdown alone is not alpha.",
        ],
        "outputs": {
            "concise_report": str(STUDY_DIR_PATH / "REPORT.md"),
            "full_report": str(STUDY_DIR_PATH / "REPORT_FULL.md"),
            "notebook": str(NOTEBOOK_PATH),
            "knowledge_record": str(STUDY_DIR_PATH / "knowledge_record.json"),
            "manifest": str(STUDY_DIR_PATH / "run_manifest.json"),
            "tables": str(TABLE_DIR_PATH),
            "charts": str(CHART_DIR_PATH),
        },
        "evidence_waivers": [
            {
                "layer": "cross-sectional IC, ranks, and PIT constituent membership",
                "reason": "The strategy trades one fixed ETF.",
                "promotion_effect": "No stock-selection claim is allowed.",
            },
            {
                "layer": "borrow and short realism",
                "reason": "Exposure is long or flat and never exceeds 100%.",
                "promotion_effect": "No effect on the research-only boundary.",
            },
            {
                "layer": "multi-position capacity aggregation",
                "reason": "There is one symbol and one sleeve in this study.",
                "promotion_effect": "SPY selected-order participation remains diagnostic only.",
            },
        ],
        "adaptive_workflow": {
            "schema_version": "quant-research-workflow-v1",
            "profile": "standard",
            "runtime_budget": {
                "target_active_minutes": 90,
                "hard_cap_active_minutes": 180,
                "max_adaptive_rounds": 2,
                "max_new_hypotheses_per_round": 4,
                "max_total_variants": 24,
                "max_parallel_lanes": 1,
            },
            "state": "research_state.json",
            "hypothesis_registry": "hypothesis_registry.json",
            "experiment_ledger": "experiment_ledger.jsonl",
            "decision_log": "decision_log.jsonl",
            "source_rule_map": "SOURCE_RULE_MAP.md",
            "holdout_policy": "Validation and confirmation are each opened once after source_exact and all 12 robustness rows are frozen.",
            "post_result_policy": "Any new rule after run time is post-hoc and may only become a future-data hypothesis.",
        },
    }
    write_json(SPEC_PATH, spec_dict)

    state_path = STUDY_DIR_PATH / "research_state.json"
    state_dict = json.loads(state_path.read_text(encoding="utf-8"))
    state_dict["phase"] = "literal_baseline"
    state_dict["evidence_phase"] = "literal_baseline"
    state_dict["updated_at"] = frozen_at_str
    state_dict["source"]["read_complete"] = True
    state_dict["locks"] = {
        "source_rule_map_frozen_at": frozen_at_str,
        "literal_baseline_frozen_at": frozen_at_str,
        "validation_locked_at": frozen_at_str,
        "confirmation_locked_at": frozen_at_str,
    }
    state_dict["holdouts"] = {
        "validation_period": f"{VALIDATION_START_STR} to {VALIDATION_END_STR}",
        "validation_opened_at": None,
        "confirmation_period": f"{CONFIRMATION_START_STR} to latest usable",
        "confirmation_opened_at": None,
    }
    state_dict["adaptive_search"]["declared_total_variants"] = (
        len(RESEARCH_VARIANT_LIST) + 3
    )
    write_json(state_path, state_dict)

    hypothesis_path = STUDY_DIR_PATH / "hypothesis_registry.json"
    hypothesis_dict = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    hypothesis_dict["updated_at"] = frozen_at_str
    hypothesis_dict["hypotheses"] = [
        {
            "hypothesis_id": "H0",
            "title": "Literal Varadi SPY Adaptive Momentum baseline",
            "provenance": "source_literal",
            "family": "baseline",
            "role": "signal",
            "classification": "baseline",
            "economic_mechanism": "Drawdown severity accelerates the trend filter during crashes and rebounds.",
            "expected_direction": "Higher Sharpe than static trend baselines with faster rebound entry.",
            "falsifier": "The executable central-cost rule does not beat static trend Sharpe in both source-unseen periods or fails the risk/return gates.",
            "created_at": state_dict["created_at"],
            "frozen_at": frozen_at_str,
            "data_periods_seen": ["Source disclosed 1995-2020"],
            "declared_variant_count": 1,
            "experiment_ids": [],
            "status": "planned",
            "evidence_summary": "not_assessed",
            "disposition": "unresolved",
        },
        {
            "hypothesis_id": "H1",
            "title": "One-factor parameter neighborhood is stable",
            "provenance": "predeclared",
            "family": "parameter_robustness",
            "role": "validation",
            "classification": "validation",
            "economic_mechanism": "A genuine gain-scheduling mechanism should not depend on one narrow endpoint or filter length.",
            "expected_direction": "Most one-factor rows keep the source rule's direction across validation and confirmation.",
            "falsifier": "Performance is isolated to source_exact or parameter ranks reverse sharply out of sample.",
            "created_at": frozen_at_str,
            "frozen_at": frozen_at_str,
            "data_periods_seen": ["Source disclosed 1995-2020"],
            "declared_variant_count": len(RESEARCH_VARIANT_LIST) - 1,
            "experiment_ids": [],
            "status": "planned",
            "evidence_summary": "not_assessed",
            "disposition": "unresolved",
        },
        {
            "hypothesis_id": "H2",
            "title": "Null explanation: beta reduction explains the result",
            "provenance": "predeclared",
            "family": "competing_explanation",
            "role": "diagnostic",
            "classification": "diagnostic",
            "economic_mechanism": "Cash timing mechanically lowers volatility and drawdown without adding timing alpha.",
            "expected_direction": "CAGR retention, beta, exposure, and benchmark-relative Sharpe reveal whether the benefit is only de-risking.",
            "falsifier": "Source_exact beats static rules and SPY on risk-adjusted evidence in both source-unseen periods after costs.",
            "created_at": frozen_at_str,
            "frozen_at": frozen_at_str,
            "data_periods_seen": [],
            "declared_variant_count": 1,
            "experiment_ids": [],
            "status": "planned",
            "evidence_summary": "not_assessed",
            "disposition": "unresolved",
        },
    ]
    write_json(hypothesis_path, hypothesis_dict)


def amend_capacity_data_contract() -> None:
    if not SPEC_PATH.exists() or not DATA_METADATA_PATH.exists():
        raise FileNotFoundError("Frozen specification and refreshed data metadata are required.")
    metadata_dict = json.loads(DATA_METADATA_PATH.read_text(encoding="utf-8"))
    if not metadata_dict.get("common_columns_unchanged", False):
        raise RuntimeError("Cannot amend capacity data without exact common-column parity.")
    spec_dict = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    old_content_id_str = str(spec_dict["data"]["input_artifacts"][0]["content_id"])
    new_content_id_str = f"sha256:{metadata_dict['cache_sha256']}"
    if old_content_id_str == new_content_id_str:
        return
    spec_dict["data"]["input_artifacts"][0]["content_id"] = new_content_id_str
    spec_dict["data"]["input_artifacts"][0]["capacity_only_added_field"] = (
        "spy_volume; all previously frozen columns verified exactly unchanged"
    )
    amendment_list = spec_dict.setdefault("amendments", [])
    amendment_list.append(
        {
            "amended_at": now_iso_str(),
            "reason": (
                "Post-result audit found Norgate Turnover was not a valid dollar-ADV input. "
                "Added raw Volume so capacity uses lagged Unadjusted Close times Volume."
            ),
            "scope": (
                "Capacity input and capacity table only; signal, target path, prices, returns, "
                "timing, costs, periods, variants, and promotion gates are unchanged."
            ),
            "old_data_content_id": old_content_id_str,
            "new_data_content_id": new_content_id_str,
            "common_columns_unchanged": True,
            "result_seen_before_amendment": True,
            "promotion_effect": "None; capacity remains diagnostic and uncalibrated.",
        }
    )
    write_json(SPEC_PATH, spec_dict)


def amend_timing_attribution_contract() -> None:
    """Add the post-review executable overnight decomposition transparently."""

    if not SPEC_PATH.exists():
        raise FileNotFoundError("Frozen specification is required.")
    spec_dict = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    old_family_str = (
        "Three timing paths and three fixed cost layers as evaluation "
        "dimensions, not selection axes"
    )
    new_family_str = (
        "Five named timing diagnostics and three fixed cost layers as "
        "evaluation dimensions, not selection axes"
    )
    declared_family_list = spec_dict["search_space"]["declared_families"]
    if new_family_str in declared_family_list:
        return
    if old_family_str not in declared_family_list:
        raise RuntimeError("Frozen timing-family declaration was not recognized.")
    declared_family_list[declared_family_list.index(old_family_str)] = new_family_str
    spec_dict.setdefault("amendments", []).append(
        {
            "amended_at": now_iso_str(),
            "reason": (
                "Independent quant review found the saved timing table omitted "
                "the tradable Close_(T+1) to Open_(T+2) overnight leg."
            ),
            "scope": (
                "Timing attribution only: add a separately labeled pre-fill "
                "overnight diagnostic and held-overnight executable leg."
            ),
            "result_seen_before_amendment": True,
            "promotion_effect": (
                "None; primary next-open returns, costs, variants, periods, "
                "promotion gates, and verdict are unchanged."
            ),
        }
    )
    write_json(SPEC_PATH, spec_dict)


def period_date_dict(cache_df: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    last_usable_decision_ts = pd.Timestamp(cache_df.index[-3])
    return {
        "source_discovery": (pd.Timestamp(EVALUATION_START_STR), pd.Timestamp(SOURCE_SAMPLE_END_STR)),
        "validation": (pd.Timestamp(VALIDATION_START_STR), pd.Timestamp(VALIDATION_END_STR)),
        "confirmation": (pd.Timestamp(CONFIRMATION_START_STR), last_usable_decision_ts),
        "full": (pd.Timestamp(EVALUATION_START_STR), last_usable_decision_ts),
    }


def compute_variant_signal_df(
    total_return_close_ser: pd.Series,
    variant_obj: ResearchVariant,
) -> pd.DataFrame:
    return compute_spy_adaptive_momentum_signal_df(
        signal_price_close_ser=total_return_close_ser,
        percentile_lookback_int=variant_obj.percentile_lookback_int,
        high_lookback_int=variant_obj.high_lookback_int,
        percentile_method_str=variant_obj.percentile_method_str,
        fast_lookback_int=variant_obj.fast_lookback_int,
        slow_lookback_int=variant_obj.slow_lookback_int,
        percentile_power_float=variant_obj.percentile_power_float,
        price_filter_lookback_int=variant_obj.price_filter_lookback_int,
    )


def build_target_weight_dict(cache_df: pd.DataFrame) -> dict[str, pd.Series]:
    total_return_close_ser = cache_df["spy_total_return_close"]
    target_weight_dict = {
        variant_obj.variant_key_str: compute_variant_signal_df(
            total_return_close_ser,
            variant_obj,
        )["target_weight_ser"]
        for variant_obj in RESEARCH_VARIANT_LIST
    }
    # *** CRITICAL*** inclusive rolling windows end at Close_T; all three
    # targets can first be executed at Open_(T+1).
    target_weight_dict["sma_200"] = total_return_close_ser.gt(
        total_return_close_ser.rolling(200, min_periods=200).mean()
    ).astype(float)
    target_weight_dict["mom_252"] = total_return_close_ser.gt(
        total_return_close_ser.shift(252)
    ).astype(float)
    target_weight_dict["mom_120"] = total_return_close_ser.gt(
        total_return_close_ser.shift(120)
    ).astype(float)
    target_weight_dict["buy_hold"] = pd.Series(
        1.0,
        index=cache_df.index,
        dtype=float,
    )
    return target_weight_dict


def build_timing_return_df(cache_df: pd.DataFrame) -> pd.DataFrame:
    total_return_open_ser = cache_df["spy_total_return_open"]
    total_return_close_ser = cache_df["spy_total_return_close"]
    # *** CRITICAL*** forward returns are labels for research evaluation only.
    # Every target at Close_T is paired with the explicitly named later prices.
    return pd.DataFrame(
        {
            "source_close_to_close": total_return_close_ser.shift(-1).divide(total_return_close_ser).sub(1.0),
            "same_exit_intraday": total_return_close_ser.shift(-1).divide(total_return_open_ser.shift(-1)).sub(1.0),
            "primary_next_open": total_return_open_ser.shift(-2).divide(total_return_open_ser.shift(-1)).sub(1.0),
            "pre_fill_overnight_to_next_open": total_return_open_ser.shift(-1).divide(total_return_close_ser).sub(1.0),
            "held_overnight_to_second_open": total_return_open_ser.shift(-2).divide(total_return_close_ser.shift(-1)).sub(1.0),
        },
        index=cache_df.index,
    )


def build_path_df(
    target_weight_ser: pd.Series,
    asset_return_ser: pd.Series,
    round_trip_bps_float: float,
) -> pd.DataFrame:
    target_weight_ser = target_weight_ser.astype(float)
    prior_target_ser = target_weight_ser.shift(1).fillna(0.0)
    changed_notional_ser = target_weight_ser.sub(prior_target_ser).abs()
    one_way_cost_float = round_trip_bps_float / 2.0 / 10_000.0
    cost_ser = changed_notional_ser.mul(one_way_cost_float)
    strategy_return_ser = target_weight_ser.mul(asset_return_ser).sub(cost_ser)
    path_df = pd.DataFrame(
        {
            "target_weight": target_weight_ser,
            "changed_notional": changed_notional_ser,
            "cost": cost_ser,
            "asset_return": asset_return_ser,
            "strategy_return": strategy_return_ser,
        }
    ).replace([np.inf, -np.inf], np.nan)
    return path_df.dropna(subset=["target_weight", "asset_return", "strategy_return"])


def compound_return_float(return_ser: pd.Series) -> float:
    if return_ser.empty:
        return math.nan
    return float((1.0 + return_ser).prod() - 1.0)


def performance_metrics_dict(path_df: pd.DataFrame) -> dict[str, float | int | str]:
    if len(path_df) < 2:
        return {
            "start_date": "N/A",
            "end_date": "N/A",
            "observation_count": len(path_df),
            "CAGR": math.nan,
            "annualized_volatility": math.nan,
            "Sharpe": math.nan,
            "maximum_drawdown": math.nan,
            "Calmar": math.nan,
            "daily_market_correlation": math.nan,
            "monthly_market_correlation": math.nan,
            "market_beta": math.nan,
            "average_exposure": math.nan,
            "annualized_turnover": math.nan,
            "state_change_count": 0,
            "entry_count": 0,
            "annualized_cost_drag": math.nan,
        }
    strategy_return_ser = path_df["strategy_return"].astype(float)
    market_return_ser = path_df["asset_return"].astype(float)
    observation_count_int = len(path_df)
    year_count_float = observation_count_int / ANNUALIZATION_FLOAT
    terminal_growth_float = float((1.0 + strategy_return_ser).prod())
    cagr_float = terminal_growth_float ** (1.0 / year_count_float) - 1.0
    annualized_volatility_float = float(
        strategy_return_ser.std(ddof=1) * np.sqrt(ANNUALIZATION_FLOAT)
    )
    sharpe_float = (
        float(strategy_return_ser.mean() / strategy_return_ser.std(ddof=1) * np.sqrt(ANNUALIZATION_FLOAT))
        if strategy_return_ser.std(ddof=1) > 0.0
        else math.nan
    )
    equity_ser = (1.0 + strategy_return_ser).cumprod()
    drawdown_ser = equity_ser.divide(equity_ser.cummax()).sub(1.0)
    maximum_drawdown_float = float(drawdown_ser.min())
    calmar_float = (
        cagr_float / abs(maximum_drawdown_float)
        if maximum_drawdown_float < 0.0
        else math.nan
    )
    daily_corr_float = float(strategy_return_ser.corr(market_return_ser))
    market_variance_float = float(market_return_ser.var(ddof=1))
    beta_float = (
        float(strategy_return_ser.cov(market_return_ser) / market_variance_float)
        if market_variance_float > 0.0
        else math.nan
    )
    monthly_strategy_ser = strategy_return_ser.groupby(
        strategy_return_ser.index.to_period("M")
    ).apply(compound_return_float)
    monthly_market_ser = market_return_ser.groupby(
        market_return_ser.index.to_period("M")
    ).apply(compound_return_float)
    monthly_corr_float = (
        float(monthly_strategy_ser.corr(monthly_market_ser))
        if len(monthly_strategy_ser) >= 2 and len(monthly_market_ser) >= 2
        else math.nan
    )
    target_weight_ser = path_df["target_weight"].astype(float)
    state_change_count_int = int(target_weight_ser.diff().abs().gt(0.0).sum())
    entry_count_int = int(target_weight_ser.diff().gt(0.0).sum())
    annualized_cost_drag_float = float(path_df["cost"].mean() * ANNUALIZATION_FLOAT)
    return {
        "start_date": path_df.index.min().date().isoformat(),
        "end_date": path_df.index.max().date().isoformat(),
        "observation_count": observation_count_int,
        "CAGR": cagr_float,
        "annualized_volatility": annualized_volatility_float,
        "Sharpe": sharpe_float,
        "maximum_drawdown": maximum_drawdown_float,
        "Calmar": calmar_float,
        "daily_market_correlation": daily_corr_float,
        "monthly_market_correlation": monthly_corr_float,
        "market_beta": beta_float,
        "average_exposure": float(target_weight_ser.mean()),
        "annualized_turnover": float(path_df["changed_notional"].mean() * ANNUALIZATION_FLOAT),
        "state_change_count": state_change_count_int,
        "entry_count": entry_count_int,
        "annualized_cost_drag": annualized_cost_drag_float,
    }


def slice_path_df(
    path_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    return path_df.loc[(path_df.index >= start_ts) & (path_df.index <= end_ts)]


def build_baseline_metrics_df(
    cache_df: pd.DataFrame,
    target_weight_dict: dict[str, pd.Series],
    timing_return_df: pd.DataFrame,
) -> pd.DataFrame:
    model_key_list = ["buy_hold", "sma_200", "mom_252", "mom_120", "source_exact"]
    period_dict = period_date_dict(cache_df)
    record_list: list[dict[str, object]] = []
    for model_key_str in model_key_list:
        for timing_key_str in ["source_close_to_close", "same_exit_intraday", "primary_next_open"]:
            for cost_layer_str, round_trip_bps_float in COST_ROUND_TRIP_BPS_DICT.items():
                effective_cost_float = 0.0 if model_key_str == "buy_hold" else round_trip_bps_float
                full_path_df = build_path_df(
                    target_weight_ser=target_weight_dict[model_key_str],
                    asset_return_ser=timing_return_df[timing_key_str],
                    round_trip_bps_float=effective_cost_float,
                )
                for period_key_str, (start_ts, end_ts) in period_dict.items():
                    metric_dict = performance_metrics_dict(
                        slice_path_df(full_path_df, start_ts, end_ts)
                    )
                    record_list.append(
                        {
                            "model": model_key_str,
                            "timing": timing_key_str,
                            "cost_layer": cost_layer_str,
                            "round_trip_bps": effective_cost_float,
                            "period": period_key_str,
                            **metric_dict,
                        }
                    )
    return pd.DataFrame(record_list)


def build_variant_metrics_df(
    cache_df: pd.DataFrame,
    target_weight_dict: dict[str, pd.Series],
    timing_return_df: pd.DataFrame,
) -> pd.DataFrame:
    period_dict = period_date_dict(cache_df)
    record_list: list[dict[str, object]] = []
    for variant_obj in RESEARCH_VARIANT_LIST:
        full_path_df = build_path_df(
            target_weight_ser=target_weight_dict[variant_obj.variant_key_str],
            asset_return_ser=timing_return_df["primary_next_open"],
            round_trip_bps_float=COST_ROUND_TRIP_BPS_DICT["central_research"],
        )
        for period_key_str, (start_ts, end_ts) in period_dict.items():
            metric_dict = performance_metrics_dict(
                slice_path_df(full_path_df, start_ts, end_ts)
            )
            record_list.append(
                {
                    **asdict(variant_obj),
                    "period": period_key_str,
                    **metric_dict,
                }
            )
    return pd.DataFrame(record_list)


def build_timing_attribution_df(
    cache_df: pd.DataFrame,
    target_weight_ser: pd.Series,
    timing_return_df: pd.DataFrame,
) -> pd.DataFrame:
    period_dict = period_date_dict(cache_df)
    record_list: list[dict[str, object]] = []
    timing_key_list = [
        "source_close_to_close",
        "pre_fill_overnight_to_next_open",
        "same_exit_intraday",
        "held_overnight_to_second_open",
        "primary_next_open",
    ]
    for timing_key_str in timing_key_list:
        full_path_df = build_path_df(
            target_weight_ser=target_weight_ser,
            asset_return_ser=timing_return_df[timing_key_str],
            round_trip_bps_float=0.0,
        )
        for period_key_str, (start_ts, end_ts) in period_dict.items():
            metric_dict = performance_metrics_dict(
                slice_path_df(full_path_df, start_ts, end_ts)
            )
            record_list.append(
                {
                    "timing": timing_key_str,
                    "period": period_key_str,
                    **metric_dict,
                }
            )
    return pd.DataFrame(record_list)


def build_state_evidence_df(
    cache_df: pd.DataFrame,
    source_signal_df: pd.DataFrame,
    timing_return_df: pd.DataFrame,
) -> pd.DataFrame:
    evidence_df = pd.DataFrame(
        {
            "target_weight": source_signal_df["target_weight_ser"],
            "severity_percentile": source_signal_df["drawdown_percentile_ser"],
            "adaptive_alpha": source_signal_df["adaptive_alpha_ser"],
            "forward_open_return": timing_return_df["primary_next_open"],
        }
    ).dropna()
    evidence_df["severity_quartile"] = pd.cut(
        evidence_df["severity_percentile"],
        bins=[-1e-12, 0.25, 0.50, 0.75, 1.0],
        labels=["q1_low", "q2", "q3", "q4_high"],
        include_lowest=True,
    )
    period_dict = period_date_dict(cache_df)
    record_list: list[dict[str, object]] = []
    for period_key_str, (start_ts, end_ts) in period_dict.items():
        period_df = evidence_df.loc[
            (evidence_df.index >= start_ts) & (evidence_df.index <= end_ts)
        ]
        for target_weight_float in [0.0, 1.0]:
            for severity_quartile_str in ["q1_low", "q2", "q3", "q4_high"]:
                cell_df = period_df.loc[
                    period_df["target_weight"].eq(target_weight_float)
                    & period_df["severity_quartile"].astype(str).eq(severity_quartile_str)
                ]
                record_list.append(
                    {
                        "period": period_key_str,
                        "regime": "risk_on" if target_weight_float == 1.0 else "risk_off",
                        "severity_quartile": severity_quartile_str,
                        "observation_count": len(cell_df),
                        "mean_forward_open_return": float(cell_df["forward_open_return"].mean()) if len(cell_df) else math.nan,
                        "annualized_mean_return": float(cell_df["forward_open_return"].mean() * ANNUALIZATION_FLOAT) if len(cell_df) else math.nan,
                        "mean_adaptive_alpha": float(cell_df["adaptive_alpha"].mean()) if len(cell_df) else math.nan,
                    }
                )
    return pd.DataFrame(record_list)


def build_capacity_df(
    cache_df: pd.DataFrame,
    target_weight_ser: pd.Series,
) -> pd.DataFrame:
    raw_dollar_volume_ser = cache_df["spy_unadjusted_close"].mul(
        cache_df["spy_volume"]
    )
    # *** CRITICAL*** decision-time liquidity uses only the prior 20 observed
    # sessions; the current session's final price and volume are excluded.
    prior_adv_ser = raw_dollar_volume_ser.shift(1).rolling(
        20,
        min_periods=20,
    ).mean()
    changed_target_ser = target_weight_ser.diff().abs()
    selected_order_df = pd.DataFrame(
        {"changed_target": changed_target_ser, "prior_adv": prior_adv_ser}
    ).dropna()
    selected_order_df = selected_order_df.loc[
        selected_order_df["changed_target"].gt(0.0)
        & selected_order_df["prior_adv"].gt(0.0)
    ]
    record_list: list[dict[str, object]] = []
    for aum_float in [1_000_000.0, 10_000_000.0, 100_000_000.0, 500_000_000.0]:
        participation_ser = selected_order_df["changed_target"].mul(aum_float).divide(
            selected_order_df["prior_adv"]
        )
        p99_float = float(participation_ser.quantile(0.99))
        if p99_float <= 0.001:
            capacity_label_str = "comfortable_research_scale"
        elif p99_float <= 0.01:
            capacity_label_str = "soft_capacity"
        elif p99_float <= 0.05:
            capacity_label_str = "strained_region"
        else:
            capacity_label_str = "hard_capacity_stress"
        record_list.append(
            {
                "AUM": aum_float,
                "order_count": len(participation_ser),
                "median_participation": float(participation_ser.median()),
                "p90_participation": float(participation_ser.quantile(0.90)),
                "p99_participation": p99_float,
                "maximum_participation": float(participation_ser.max()),
                "capacity_label": capacity_label_str,
                "limitation": "Daily turnover is not opening-auction volume; no impact is calibrated.",
            }
        )
    return pd.DataFrame(record_list)


def dataframe_markdown_str(dataframe_obj: pd.DataFrame) -> str:
    column_list = [str(column_obj) for column_obj in dataframe_obj.columns]

    def format_cell_str(value_obj: object) -> str:
        if isinstance(value_obj, (float, np.floating)):
            if not np.isfinite(float(value_obj)):
                return "N/A"
            return f"{float(value_obj):.3f}"
        if isinstance(value_obj, (bool, np.bool_)):
            return "yes" if bool(value_obj) else "no"
        return str(value_obj).replace("|", "\\|").replace("\n", " ")

    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join("---" for _ in column_list) + " |"
    row_str_list = [
        "| "
        + " | ".join(format_cell_str(value_obj) for value_obj in row_tuple)
        + " |"
        for row_tuple in dataframe_obj.itertuples(index=False, name=None)
    ]
    return "\n".join([header_str, separator_str, *row_str_list])


def evaluate_verdict_dict(baseline_metrics_df: pd.DataFrame) -> dict[str, object]:
    primary_df = baseline_metrics_df.loc[
        baseline_metrics_df["timing"].eq("primary_next_open")
        & baseline_metrics_df["cost_layer"].eq("central_research")
    ].copy()
    gate_record_list: list[dict[str, object]] = []
    for period_key_str in ["validation", "confirmation"]:
        period_df = primary_df.loc[primary_df["period"].eq(period_key_str)].set_index("model")
        adaptive_ser = period_df.loc["source_exact"]
        buy_hold_ser = period_df.loc["buy_hold"]
        best_static_sharpe_float = float(
            period_df.loc[["sma_200", "mom_252", "mom_120"], "Sharpe"].max()
        )
        cagr_retention_float = (
            float(adaptive_ser["CAGR"] / buy_hold_ser["CAGR"])
            if buy_hold_ser["CAGR"] > 0.0
            else math.nan
        )
        drawdown_ratio_float = (
            float(abs(adaptive_ser["maximum_drawdown"]) / abs(buy_hold_ser["maximum_drawdown"]))
            if buy_hold_ser["maximum_drawdown"] < 0.0
            else math.nan
        )
        gate_record_list.extend(
            [
                {
                    "period": period_key_str,
                    "gate": "CAGR_retention_at_least_70pct",
                    "value": cagr_retention_float,
                    "threshold": 0.70,
                    "passed": bool(cagr_retention_float >= 0.70),
                },
                {
                    "period": period_key_str,
                    "gate": "max_drawdown_ratio_at_most_80pct",
                    "value": drawdown_ratio_float,
                    "threshold": 0.80,
                    "passed": bool(drawdown_ratio_float <= 0.80),
                },
                {
                    "period": period_key_str,
                    "gate": "Sharpe_at_least_best_static",
                    "value": float(adaptive_ser["Sharpe"] - best_static_sharpe_float),
                    "threshold": 0.0,
                    "passed": bool(adaptive_ser["Sharpe"] >= best_static_sharpe_float),
                },
            ]
        )
    conservative_full_ser = baseline_metrics_df.loc[
        baseline_metrics_df["model"].eq("source_exact")
        & baseline_metrics_df["timing"].eq("primary_next_open")
        & baseline_metrics_df["cost_layer"].eq("conservative_survival")
        & baseline_metrics_df["period"].eq("full")
    ].iloc[0]
    gate_record_list.append(
        {
            "period": "full",
            "gate": "conservative_CAGR_positive",
            "value": float(conservative_full_ser["CAGR"]),
            "threshold": 0.0,
            "passed": bool(conservative_full_ser["CAGR"] > 0.0),
        }
    )
    gate_df = pd.DataFrame(gate_record_list)
    all_passed_bool = bool(gate_df["passed"].all())
    return {
        "all_passed": all_passed_bool,
        "research_status": "research_candidate" if all_passed_bool else "diagnostic",
        "disposition": "candidate" if all_passed_bool else "diagnostic",
        "verdict": (
            "PASS_RESEARCH_ONLY: the literal adaptive regime passed every frozen historical gate, but remains outside PAPER/LIVE."
            if all_passed_bool
            else "FAIL_FROZEN_GATES: the literal adaptive regime failed at least one source-unseen return, drawdown, or static-baseline gate; retain it only as a diagnostic market-state overlay."
        ),
        "gate_df": gate_df,
    }


def replication_outcome_dict(baseline_metrics_df: pd.DataFrame) -> dict[str, object]:
    local_ser = baseline_metrics_df.loc[
        baseline_metrics_df["model"].eq("source_exact")
        & baseline_metrics_df["timing"].eq("source_close_to_close")
        & baseline_metrics_df["cost_layer"].eq("paper_like")
        & baseline_metrics_df["period"].eq("source_discovery")
    ].iloc[0]
    comparison_record_list = []
    tolerance_dict = {
        "CAGR": 0.02,
        "annualized_volatility": 0.02,
        "Sharpe": 0.15,
        "maximum_drawdown": 0.05,
        "state_change_count": 35.0,
    }
    for metric_str, source_value_float in SOURCE_REPORTED_METRIC_DICT.items():
        local_value_float = float(local_ser[metric_str])
        difference_float = local_value_float - source_value_float
        comparison_record_list.append(
            {
                "metric": metric_str,
                "source_reported": source_value_float,
                "local_source_like": local_value_float,
                "difference": difference_float,
                "tolerance": tolerance_dict[metric_str],
                "within_tolerance": abs(difference_float) <= tolerance_dict[metric_str],
            }
        )
    comparison_df = pd.DataFrame(comparison_record_list)
    within_count_int = int(comparison_df["within_tolerance"].sum())
    if within_count_int == len(comparison_df):
        outcome_str = "replicated"
    elif within_count_int >= 3:
        outcome_str = "directionally_replicated"
    else:
        outcome_str = "not_reproducible"
    return {"outcome": outcome_str, "comparison_df": comparison_df}


def save_tables(table_dict: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    for filename_str, dataframe_obj in table_dict.items():
        dataframe_obj.to_csv(TABLE_DIR_PATH / filename_str, index=False)


def save_charts(
    cache_df: pd.DataFrame,
    target_weight_dict: dict[str, pd.Series],
    timing_return_df: pd.DataFrame,
    variant_metrics_df: pd.DataFrame,
    source_signal_df: pd.DataFrame,
) -> None:
    CHART_DIR_PATH.mkdir(parents=True, exist_ok=True)
    color_dict = {
        "buy_hold": "#1f77b4",
        "source_exact": "#d62728",
        "sma_200": "#2ca02c",
        "mom_252": "#9467bd",
        "mom_120": "#8c564b",
    }
    fig_obj, axis_arr = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model_key_str in ["buy_hold", "source_exact", "sma_200", "mom_252", "mom_120"]:
        path_df = build_path_df(
            target_weight_dict[model_key_str],
            timing_return_df["primary_next_open"],
            0.0 if model_key_str == "buy_hold" else COST_ROUND_TRIP_BPS_DICT["central_research"],
        )
        path_df = path_df.loc[path_df.index >= pd.Timestamp(EVALUATION_START_STR)]
        equity_ser = (1.0 + path_df["strategy_return"]).cumprod()
        drawdown_ser = equity_ser.divide(equity_ser.cummax()).sub(1.0)
        axis_arr[0].plot(equity_ser.index, equity_ser, label=model_key_str, color=color_dict[model_key_str], linewidth=1.4)
        axis_arr[1].plot(drawdown_ser.index, drawdown_ser, label=model_key_str, color=color_dict[model_key_str], linewidth=1.1)
    axis_arr[0].set_yscale("log")
    axis_arr[0].set_title("SPY regime and baselines | next-open | 10 bps RT | growth of 1")
    axis_arr[0].set_ylabel("Equity (log)")
    axis_arr[0].legend(ncol=3)
    axis_arr[1].set_title("Drawdown | same sample and cost layer")
    axis_arr[1].set_ylabel("Drawdown")
    axis_arr[1].grid(alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(CHART_DIR_PATH / "primary_equity_drawdown.png", dpi=160)
    plt.close(fig_obj)

    recent_mask_ser = source_signal_df.index >= pd.Timestamp("2018-01-01")
    fig_obj, axis_obj = plt.subplots(figsize=(12, 5))
    axis_obj.plot(
        source_signal_df.index[recent_mask_ser],
        source_signal_df.loc[recent_mask_ser, "signal_price_close_ser"],
        label="SPY TR close",
        color="#1f77b4",
        linewidth=1.0,
    )
    axis_obj.plot(
        source_signal_df.index[recent_mask_ser],
        source_signal_df.loc[recent_mask_ser, "adaptive_moving_average_ser"],
        label="Adaptive EMA",
        color="#d62728",
        linewidth=1.2,
    )
    risk_off_ser = source_signal_df.loc[recent_mask_ser, "target_weight_ser"].eq(0.0)
    axis_obj.fill_between(
        source_signal_df.index[recent_mask_ser],
        0.0,
        1.0,
        where=risk_off_ser.to_numpy(),
        transform=axis_obj.get_xaxis_transform(),
        color="#999999",
        alpha=0.18,
        label="Risk off",
    )
    axis_obj.set_title("Literal Adaptive Momentum state | 2018-latest | Close_T signal")
    axis_obj.set_ylabel("Total-return adjusted level")
    axis_obj.legend()
    axis_obj.grid(alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(CHART_DIR_PATH / "adaptive_state_recent.png", dpi=160)
    plt.close(fig_obj)

    central_path_df = build_path_df(
        target_weight_dict["source_exact"],
        timing_return_df["primary_next_open"],
        COST_ROUND_TRIP_BPS_DICT["central_research"],
    )
    # *** CRITICAL*** rolling dependence is trailing and uses only observations
    # already realized through each plotted date.
    rolling_corr_ser = central_path_df["strategy_return"].rolling(126, min_periods=126).corr(
        central_path_df["asset_return"]
    )
    fig_obj, axis_obj = plt.subplots(figsize=(12, 4.5))
    axis_obj.plot(rolling_corr_ser.index, rolling_corr_ser, color="#ff7f0e", linewidth=1.0)
    axis_obj.axhline(0.0, color="black", linewidth=0.8)
    axis_obj.set_title("126-session rolling SPY correlation | next-open | 10 bps RT")
    axis_obj.set_ylabel("Correlation")
    axis_obj.grid(alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(CHART_DIR_PATH / "rolling_spy_correlation.png", dpi=160)
    plt.close(fig_obj)

    pivot_df = variant_metrics_df.pivot(index="variant_key_str", columns="period", values="Sharpe")
    fig_obj, axis_obj = plt.subplots(figsize=(7, 6))
    axis_obj.scatter(pivot_df["validation"], pivot_df["confirmation"], color="#4c78a8", alpha=0.8)
    for variant_key_str, row_ser in pivot_df.iterrows():
        if variant_key_str == "source_exact":
            axis_obj.scatter(row_ser["validation"], row_ser["confirmation"], color="#d62728", s=80, zorder=3)
            axis_obj.annotate("source_exact", (row_ser["validation"], row_ser["confirmation"]), xytext=(5, 5), textcoords="offset points")
    min_axis_float = float(np.nanmin([pivot_df["validation"].min(), pivot_df["confirmation"].min()]))
    max_axis_float = float(np.nanmax([pivot_df["validation"].max(), pivot_df["confirmation"].max()]))
    axis_obj.plot([min_axis_float, max_axis_float], [min_axis_float, max_axis_float], linestyle="--", color="gray", linewidth=1.0)
    axis_obj.set_title("Frozen one-factor variants | Sharpe stability | next-open | 10 bps RT")
    axis_obj.set_xlabel("Validation Sharpe (2021-2023)")
    axis_obj.set_ylabel("Confirmation Sharpe (2024-latest)")
    axis_obj.grid(alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(CHART_DIR_PATH / "variant_validation_confirmation.png", dpi=160)
    plt.close(fig_obj)


def write_reports(
    baseline_metrics_df: pd.DataFrame,
    variant_metrics_df: pd.DataFrame,
    timing_attribution_df: pd.DataFrame,
    state_evidence_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    verdict_dict: dict[str, object],
    replication_dict: dict[str, object],
) -> None:
    primary_full_df = baseline_metrics_df.loc[
        baseline_metrics_df["timing"].eq("primary_next_open")
        & baseline_metrics_df["cost_layer"].eq("central_research")
        & baseline_metrics_df["period"].eq("full")
        & baseline_metrics_df["model"].isin(["buy_hold", "sma_200", "mom_252", "mom_120", "source_exact"])
    ].copy()
    display_column_list = [
        "model",
        "CAGR",
        "annualized_volatility",
        "Sharpe",
        "maximum_drawdown",
        "market_beta",
        "daily_market_correlation",
        "average_exposure",
        "annualized_turnover",
    ]
    display_df = primary_full_df[display_column_list].copy()
    for column_str in [
        "CAGR",
        "annualized_volatility",
        "maximum_drawdown",
        "average_exposure",
        "annualized_turnover",
    ]:
        display_df[column_str] = display_df[column_str].mul(100.0)
    display_df = display_df.rename(
        columns={
            "model": "Series",
            "annualized_volatility": "Vol %",
            "maximum_drawdown": "Max DD %",
            "market_beta": "Beta",
            "daily_market_correlation": "Daily Corr",
            "average_exposure": "Exposure %",
            "annualized_turnover": "Turnover %",
            "CAGR": "CAGR %",
        }
    )
    source_exact_ser = primary_full_df.loc[primary_full_df["model"].eq("source_exact")].iloc[0]
    confirmation_ser = baseline_metrics_df.loc[
        baseline_metrics_df["model"].eq("source_exact")
        & baseline_metrics_df["timing"].eq("primary_next_open")
        & baseline_metrics_df["cost_layer"].eq("central_research")
        & baseline_metrics_df["period"].eq("confirmation")
    ].iloc[0]
    failed_gate_count_int = int((~verdict_dict["gate_df"]["passed"]).sum())
    replication_outcome_str = str(replication_dict["outcome"])
    status_str = str(verdict_dict["research_status"])
    validation_variant_ser = variant_metrics_df.loc[
        variant_metrics_df["period"].eq("validation"), "Sharpe"
    ]
    confirmation_variant_ser = variant_metrics_df.loc[
        variant_metrics_df["period"].eq("confirmation"), "Sharpe"
    ]
    held_overnight_full_ser = timing_attribution_df.loc[
        timing_attribution_df["timing"].eq("held_overnight_to_second_open")
        & timing_attribution_df["period"].eq("full")
    ].iloc[0]
    intraday_full_ser = timing_attribution_df.loc[
        timing_attribution_df["timing"].eq("same_exit_intraday")
        & timing_attribution_df["period"].eq("full")
    ].iloc[0]

    report_str = rf"""# משטר שוק Adaptive Momentum על SPY

> **RESEARCH בלבד. אין כאן אישור PAPER, LIVE, allocation או מסחר.**

## TL;DR

האסטרטגיה מחזיקה 100% `SPY` כאשר `SMA10` של מחיר Total Return נמצא מעל EMA אדפטיבי, ועוברת ל־cash כאשר הוא מתחתיו. מהירות ה־EMA נעה בין 200 ל־50 ימים לפי חומרת ה־drawdown ביחס ל־126 sessions אחרונים. הבדיקה משתמשת ב־Norgate מ־1993 ועד האס־אוף השמור, והכלל נמדד בעיתוי סיבתי `Close_T → Open_(T+1)` עם 10 bps round-trip בשורה הראשית. שחזור המקור סווג `{replication_outcome_str}`. הכלל נכשל ב־{failed_gate_count_int} מתוך {len(verdict_dict['gate_df'])} השערים הקפואים, ולכן הסטטוס הוא `{status_str}` ולא מועמד למסחר.

## האסטרטגיה בשפה פשוטה

הרעיון הוא לא לנחש “bull” או “bear” בעזרת classifier. במקום זאת, ה־drawdown קובע כמה מהר פילטר המגמה מגיב. ליד שיא כל הזמנים הפילטר דומה ל־EMA200 ואינו נבהל מתיקון קטן. כאשר ה־drawdown חריג לעומת ששת החודשים האחרונים, הפילטר מתקרב ל־EMA50, כדי לצאת ולהיכנס מחדש מהר יותר. זהו `gain scheduling` פשוט ושקוף.

## הכללים המדויקים והעיתוי

$$
DD_T = \frac{{P_T}}{{\max_{{s \le T}} P_s}} - 1
$$

`q_T` הוא דירוג ECDF סיבתי של `-DD_T` בחלון 126 sessions, ו־`Q_T=q_T^2`.

$$
\alpha_T = Q_T \frac{{2}}{{51}} + (1-Q_T) \frac{{2}}{{201}}
$$

$$
AMA_T = \alpha_T P_T + (1-\alpha_T) AMA_{{T-1}}
$$

אם `SMA10(P)_T > AMA_T`, יעד החשיפה הוא 100% SPY; אחרת 0%. כל הקלטים ידועים רק אחרי `Close_T`. שינוי החשיפה הראשון האפשרי הוא ב־`Open_(T+1)`. תוצאת close-to-close של המקור מוצגת כאבחון בלבד. cash מקבל 0% תשואה.

```text
SPY TR Close_T -> DD percentile -> alpha_T -> AMA_T -> risk_on/off
                                                       |
                                                       | Close_T decision
                                                       v
                                                  Open_(T+1)
```

## נתונים ותכנון הבדיקה

- נכס: SPY יחיד, ללא survivorship selection.
- signal: Norgate `TOTALRETURN`; ביצוע המנוע: `CAPITALSPECIAL` עם dividend ledger נפרד.
- discovery מזוהם־מקור: 1995-2020.
- validation שלא נצפה במקור: 2021-2023.
- confirmation נעול: 2024 ועד האס־אוף השמור.
- עלויות: 0 / 10 / 25 bps round-trip על שינוי notional.
- משפחת חיפוש: הכלל המילולי ועוד 12 וריאנטים one-factor בלבד; שלושה baselines סטטיים. אין Cartesian grid ואין בחירת winner בדיעבד.
- זמן מחקר פעיל שנרשם: כ־{ACTIVE_MINUTES_USED_INT} דקות. סיבת העצירה: המשפחה הקפואה הושלמה ואין היתר ל־retuning על אותה היסטוריה.
- מונחי audit: `verdict`, `replication`, `validation`, `runtime`, `research-only`.

## תוצאות מול benchmark

השורות הבאות משתמשות באותו מדגם, `Open_(T+1) → Open_(T+2)`, וב־10 bps round-trip לכל אסטרטגיה פעילה. Sharpe מחושב עם ריבית חסרת סיכון 0.

{dataframe_markdown_str(display_df)}

![Equity and drawdown](charts/primary_equity_drawdown.png)

במדגם המלא, `source_exact` השיג CAGR של {float(source_exact_ser['CAGR']):.2%}, Sharpe של {float(source_exact_ser['Sharpe']):.3f}, ו־Max DD של {float(source_exact_ser['maximum_drawdown']):.2%}. בתקופת confirmation בלבד, Sharpe היה {float(confirmation_ser['Sharpe']):.3f}. מספרים אלה הם backtest, לא הוכחת edge.

## קשר לשוק וחשיפה

ה־beta המלא מול SPY היה {float(source_exact_ser['market_beta']):.3f}, הקורלציה היומית {float(source_exact_ser['daily_market_correlation']):.3f}, והחשיפה הממוצעת {float(source_exact_ser['average_exposure']):.1%}. לכן כל שיפור בסיכון חייב להיקרא יחד עם הזמן ב־cash; קורלציה נמוכה יותר אינה לבדה alpha.

![Rolling correlation](charts/rolling_spy_correlation.png)

## יציבות, עיתוי ועלויות

המקור אינו מגדיר fill. לכן הופרדו close-to-close מקור־דמוי, overnight שלפני המילוי שאינו סחיר לאחר החלטת `Close_T`, intraday לאחר `Open_(T+1)`, overnight שמוחזק עד `Open_(T+2)`, והיישום היומי open-to-open. כמו כן, כל 12 השכנים הקפואים מוצגים ולא רק הטוב שבהם.

במדגם המלא וללא עלות, הרגל הסחירה `Close_(T+1) → Open_(T+2)` השיגה CAGR של {float(held_overnight_full_ser['CAGR']):.2%} ו־Sharpe של {float(held_overnight_full_ser['Sharpe']):.3f}; הרגל `Open_(T+1) → Close_(T+1)` השיגה CAGR של {float(intraday_full_ser['CAGR']):.2%} ו־Sharpe של {float(intraday_full_ser['Sharpe']):.3f}. לכן התשואה של מסלול ה־open-to-open מרוכזת בעיקר ב־overnight שמוחזק בפועל, לא ב־overnight שלפני הכניסה.

ה־neighborhood אינו אחיד: Sharpe של הווריאנטים נע בין {float(validation_variant_ser.min()):.3f} ל־{float(validation_variant_ser.max()):.3f} ב־validation, ובין {float(confirmation_variant_ser.min()):.3f} ל־{float(confirmation_variant_ser.max()):.3f} ב־confirmation. לכן המעבר של `source_exact` אינו הוכחה שכל בחירת פרמטר קרובה יציבה.

![Variant stability](charts/variant_validation_confirmation.png)

![Adaptive state](charts/adaptive_state_recent.png)

## סיכונים ומגבלות

- דירוג האחוזון, ties, EMA seed, מחיר fill וספירת trades אינם מוגדרים במלואם במקור.
- 1995-2020 אינו OOS; המחבר בחר את הכלל לאחר שראה את ההיסטוריה הזאת.
- עלויות ה־open הן אומדן; אין spread, auction volume, queue או partial fills אמפיריים.
- SPY Total Return הוא proxy נקי לסיגנל, אך LIVE יצטרך לשמר במדויק dividend accounting ו־order timing.
- 13 וריאנטים ועוד שלושה baselines יוצרים multiplicity; אין להציג את השורה הטובה ביותר כאישור עצמאי.
- capacity מבוסס על daily turnover קודם, לא על opening-auction volume, ולכן הוא אבחוני בלבד.

## המלצה סופית

**{verdict_dict['verdict']}**

ההמלצה היא להשאיר את המודל ב־RESEARCH. אין לשנות את `MarketRegimeFilterStrategy` הקיים, אין לרשום את המודל ל־LIVE, ואין להקצות הון. השער הבא, אם רוצים להמשיך, הוא shadow קפוא של החלטות יומיות ו־Open fills אמפיריים ללא שינוי פרמטרים.
"""
    (STUDY_DIR_PATH / "REPORT.md").write_text(report_str, encoding="utf-8")

    full_report_str = report_str + rf"""

מונחי ביקורת החבילה: `source`, `timing`, `search`, `holdout`, `failure`, `artifact`.

## נספח א: שחזור המקור

{dataframe_markdown_str(replication_dict['comparison_df'])}

סיווג השחזור הוא `{replication_outcome_str}`. גם התאמה מספרית אינה מתקנת את קונפליקט העיתוי; close-to-close נשאר diagnostic.

## נספח ב: שערי הקידום הקפואים

{dataframe_markdown_str(verdict_dict['gate_df'])}

## נספח ג: טבלת עיתוי מלאה

{dataframe_markdown_str(timing_attribution_df)}

זהות התשואה לכל session נשמרת:

$$
1+r_{{close\_to\_close}}=(1+r_{{overnight}})(1+r_{{intraday}})
$$

וגם הזהות הסחירה של מסלול ה־headline נשמרת:

$$
1+r_{{open\_to\_open}}=(1+r_{{intraday\ T+1}})(1+r_{{held\ overnight}})
$$

## נספח ד: כל וריאנטי ה־one-factor

{dataframe_markdown_str(variant_metrics_df)}

## נספח ה: מצבי חומרת drawdown

{dataframe_markdown_str(state_evidence_df)}

## נספח ו: capacity אבחוני

{dataframe_markdown_str(capacity_df)}

## נספח ז: ביקורת כשלים כמותיים

- **Lookahead:** כל `cummax`, rolling ו־EMA מסתיימים ב־Close_T; forward returns משמשים labels בלבד.
- **Survivorship:** לא חל על ETF יחיד; אין constituent universe.
- **Data snooping:** 13 adaptive rows הוקפאו מראש; אין Cartesian grid ואין החלפת baseline במנצח.
- **In-sample contamination:** 1995-2020 מסומן discovery בלבד.
- **Target leakage:** אין target בתוך הסיגנל; רק מחירי עבר ועד Close_T.
- **Regime dependence:** validation, confirmation, drawdown quartiles ו־2018+ state chart נפרדים.
- **Sample size:** המדגם היומי גדול, אך crash/rebound states מקובצים בזמן ואינם observations בלתי־תלויים.
- **Corporate actions:** signal ו־timing returns הם Total Return; המנוע סוחר CAPITALSPECIAL ומטפל בדיבידנד בנפרד.
- **Costs and slippage:** 0/10/25 bps RT; open-specific basis risk אינו מכויל.
- **Live divergence:** אין PAPER/LIVE evidence, broker ACK, fills, reconcile או operational state.

## נספח ח: מפת ארטיפקטים

- `REPORT.md` ו־`REPORT_FULL.md`: החלטה קצרה ומלאה.
- `SOURCE_RULE_MAP.md`: כללי המקור והפערים.
- `research_spec_frozen.json`: חוזה קפוא לפני תוצאות.
- `research_state.json`, `hypothesis_registry.json`, `experiment_ledger.jsonl`, `decision_log.jsonl`: lineage אדפטיבי.
- `tables/`: מספרי המקור, baselines, וריאנטים, עיתוי, states, costs ו־capacity.
- `charts/`: equity/drawdown, state, rolling correlation ויציבות.
- `{NOTEBOOK_PATH.name}`: notebook החלטה executed ללא error cells.
- `run_manifest.json`: hashes של כל הארטיפקטים המהותיים.
"""
    (STUDY_DIR_PATH / "REPORT_FULL.md").write_text(
        full_report_str,
        encoding="utf-8",
    )


def write_knowledge_record(
    baseline_metrics_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    verdict_dict: dict[str, object],
    replication_dict: dict[str, object],
) -> None:
    primary_ser = baseline_metrics_df.loc[
        baseline_metrics_df["model"].eq("source_exact")
        & baseline_metrics_df["timing"].eq("primary_next_open")
        & baseline_metrics_df["cost_layer"].eq("central_research")
        & baseline_metrics_df["period"].eq("full")
    ].iloc[0]
    timing_ser = baseline_metrics_df.loc[
        baseline_metrics_df["model"].eq("source_exact")
        & baseline_metrics_df["timing"].eq("source_close_to_close")
        & baseline_metrics_df["cost_layer"].eq("paper_like")
        & baseline_metrics_df["period"].eq("full")
    ].iloc[0]
    knowledge_record_dict = {
        "schema_version": "quant-research-knowledge-v1",
        "study_id": STUDY_ID_STR,
        "title": "SPY Adaptive Momentum Market Regime",
        "created_at": now_iso_str(),
        "last_reviewed_at": now_iso_str(),
        "research_status": verdict_dict["research_status"],
        "disposition": verdict_dict["disposition"],
        "replication_outcome": replication_dict["outcome"],
        "signal_family": "adaptive_time_series_momentum_regime",
        "objective": "Test Varadi's drawdown-adaptive SPY regime under causal next-open timing, costs, static baselines, and source-unseen periods.",
        "verdict": verdict_dict["verdict"],
        "verdicts": {
            "source_replication": replication_dict["outcome"],
            "predictive_value": "Binary regime return separation is diagnostic; see state_evidence.csv.",
            "economic_value": verdict_dict["verdict"],
            "promotion": f"{verdict_dict['research_status']}; never PAPER/LIVE authority.",
        },
        "universes": ["Fixed SPY ETF"],
        "decision_timing": "After Close_T",
        "fill_timing": "Open_T+1",
        "timing_attribution": {
            "status": "tested",
            "diagnostic_path": "Close_T to Close_T+1, including pre-fill overnight",
            "executable_path": "Open_T+1 to Close_T+1 plus held overnight Close_T+1 to Open_T+2",
            "method": "Exact compounded decomposition of both source-like close-to-close and primary open-to-open returns",
            "headline_result": f"Source-like full Sharpe {float(timing_ser['Sharpe']):.3f}; the held-overnight executable leg is reported separately in timing_attribution.csv.",
            "metrics": {},
            "artifact": "pakal-research/reports/spy_adaptive_momentum_regime_study/tables/timing_attribution.csv",
        },
        "primary_cost_layer": "central_research",
        "primary_metrics": {
            "period": f"{primary_ser['start_date']} to {primary_ser['end_date']}",
            "universe": "Fixed SPY ETF",
            "cost_layer": "central_research",
            "CAGR": float(primary_ser["CAGR"]),
            "annualized_volatility": float(primary_ser["annualized_volatility"]),
            "Sharpe": float(primary_ser["Sharpe"]),
            "maximum_drawdown": float(primary_ser["maximum_drawdown"]),
            "turnover": float(primary_ser["annualized_turnover"]),
        },
        "feature_findings": [
            {
                "feature": "Drawdown-severity adaptive EMA speed",
                "role": "market regime / risk overlay",
                "direction": "larger relative drawdown increases alpha toward EMA50",
                "status": verdict_dict["research_status"],
                "effect_size": f"Full next-open central Sharpe {float(primary_ser['Sharpe']):.3f}",
                "period_consistency": "See validation and confirmation rows in baseline_metrics.csv and variant_metrics.csv.",
                "corrected_significance": "Not claimed; full frozen family is reported without winner promotion.",
                "economic_mechanism": "Faster response during crashes and rebounds.",
                "recommended_action": "Keep research-only; no registry or LIVE wiring.",
            }
        ],
        "cost_capacity": {
            "paper_like_round_trip_bps": 0.0,
            "central_research_round_trip_bps": 10.0,
            "conservative_survival_round_trip_bps": 25.0,
            "capacity_impact_separate": True,
            "comfortable_capacity": capacity_df.loc[capacity_df["capacity_label"].eq("comfortable_research_scale"), "AUM"].tolist(),
            "soft_capacity": capacity_df.loc[capacity_df["capacity_label"].eq("soft_capacity"), "AUM"].tolist(),
            "strained_capacity": capacity_df.loc[capacity_df["capacity_label"].eq("strained_region"), "AUM"].tolist(),
            "hard_capacity": capacity_df.loc[capacity_df["capacity_label"].eq("hard_capacity_stress"), "AUM"].tolist(),
            "unresolved_reason": "Daily turnover is not opening-auction volume and impact is not calibrated.",
        },
        "limitations": [
            "Source fill, costs, percentile ties, seed, and exact sample are incomplete.",
            "1995-2020 is source-contaminated discovery.",
            "Open auction execution and impact are unmeasured.",
            "Lower beta from cash time is not alpha proof.",
        ],
        "next_tests": [
            "Freeze daily decisions in a forward shadow without changing parameters.",
            "Measure actual open spreads, fills, and basis risk before any PAPER review.",
        ],
        "sources": [str(SOURCE_PART_1_PATH), str(SOURCE_PART_2_PATH)],
        "artifacts": {
            "concise_report": "pakal-research/reports/spy_adaptive_momentum_regime_study/REPORT.md",
            "full_report": "pakal-research/reports/spy_adaptive_momentum_regime_study/REPORT_FULL.md",
            "notebook": "pakal-research/spy_adaptive_momentum_regime_study.ipynb",
            "frozen_specification": "pakal-research/reports/spy_adaptive_momentum_regime_study/research_spec_frozen.json",
            "manifest": "pakal-research/reports/spy_adaptive_momentum_regime_study/run_manifest.json",
            "primary_source_code": [
                "pakal-research/spy_adaptive_momentum_regime_study.ipynb",
            ],
            "primary_tables": [
                "pakal-research/reports/spy_adaptive_momentum_regime_study/tables/baseline_metrics.csv",
                "pakal-research/reports/spy_adaptive_momentum_regime_study/tables/variant_metrics.csv",
                "pakal-research/reports/spy_adaptive_momentum_regime_study/tables/timing_attribution.csv",
            ],
            "primary_charts": [
                "pakal-research/reports/spy_adaptive_momentum_regime_study/charts/primary_equity_drawdown.png",
                "pakal-research/reports/spy_adaptive_momentum_regime_study/charts/rolling_spy_correlation.png",
                "pakal-research/reports/spy_adaptive_momentum_regime_study/charts/variant_validation_confirmation.png",
            ],
            "research_state": "pakal-research/reports/spy_adaptive_momentum_regime_study/research_state.json",
            "hypothesis_registry": "pakal-research/reports/spy_adaptive_momentum_regime_study/hypothesis_registry.json",
            "experiment_ledger": "pakal-research/reports/spy_adaptive_momentum_regime_study/experiment_ledger.jsonl",
            "decision_log": "pakal-research/reports/spy_adaptive_momentum_regime_study/decision_log.jsonl",
            "source_rule_map": "pakal-research/reports/spy_adaptive_momentum_regime_study/SOURCE_RULE_MAP.md",
        },
        "adaptive_lineage": {
            "profile": "standard",
            "rounds_completed": 1,
            "declared_total_variants": len(RESEARCH_VARIANT_LIST) + 3,
            "actual_total_variants": len(RESEARCH_VARIANT_LIST) + 3,
            "active_minutes_used": ACTIVE_MINUTES_USED_INT,
            "stop_reason": "Frozen source baseline and one-factor robustness family fully evaluated; no historical retuning permitted.",
        },
        "tags": ["research-only", "SPY", "market-regime", "adaptive-momentum", "next-open"],
    }
    write_json(STUDY_DIR_PATH / "knowledge_record.json", knowledge_record_dict)


def write_executed_notebook() -> None:
    notebook_obj = nbformat.v4.new_notebook()
    notebook_obj["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# SPY Adaptive Momentum Market Regime\n\nRESEARCH בלבד. This decision notebook reads immutable saved evidence; it does not rerun Norgate or the strategy search."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nstudy_dir_path = Path('pakal-research/reports/spy_adaptive_momentum_regime_study')\nbaseline_df = pd.read_csv(study_dir_path / 'tables/baseline_metrics.csv')\nvariant_df = pd.read_csv(study_dir_path / 'tables/variant_metrics.csv')\nbaseline_df.query(\"timing == 'primary_next_open' and cost_layer == 'central_research' and period == 'full'\")"
        ),
        nbformat.v4.new_code_cell(
            "variant_df.pivot(index='variant_key_str', columns='period', values='Sharpe').sort_values('confirmation', ascending=False)"
        ),
        nbformat.v4.new_code_cell(
            "from IPython.display import Image, display\ndisplay(Image(filename=str(study_dir_path / 'charts/primary_equity_drawdown.png')))\ndisplay(Image(filename=str(study_dir_path / 'charts/variant_validation_confirmation.png')))"
        ),
    ]
    notebook_obj["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    executor_obj = ExecutePreprocessor(timeout=120, kernel_name="python3")
    executor_obj.preprocess(
        notebook_obj,
        {"metadata": {"path": str(PAKAL_ROOT_PATH)}},
    )
    nbformat.write(notebook_obj, NOTEBOOK_PATH)


def update_final_state(
    verdict_dict: dict[str, object],
    replication_dict: dict[str, object],
) -> None:
    state_path = STUDY_DIR_PATH / "research_state.json"
    state_dict = json.loads(state_path.read_text(encoding="utf-8"))
    finished_at_str = now_iso_str()
    state_dict["phase"] = "closeout"
    state_dict["evidence_phase"] = "confirmation"
    state_dict["updated_at"] = finished_at_str
    state_dict["runtime_budget"]["active_minutes_used"] = ACTIVE_MINUTES_USED_INT
    state_dict["baseline"] = {
        "status": "completed",
        "replication_outcome": replication_dict["outcome"],
        "executable_translation": "completed",
        "primary_evidence": [
            "tables/baseline_metrics.csv",
            "tables/timing_attribution.csv",
        ],
    }
    state_dict["holdouts"]["validation_opened_at"] = finished_at_str
    state_dict["holdouts"]["confirmation_opened_at"] = finished_at_str
    state_dict["adaptive_search"] = {
        "rounds_completed": 1,
        "actual_new_hypotheses_by_round": [0],
        "declared_total_variants": len(RESEARCH_VARIANT_LIST) + 3,
        "actual_total_variants": len(RESEARCH_VARIANT_LIST) + 3,
        "stop_reason": "Completed frozen one-factor family; no post-result retuning permitted.",
    }
    state_dict["final_decision"] = {
        "disposition": verdict_dict["disposition"],
        "research_status": verdict_dict["research_status"],
        "verdict": verdict_dict["verdict"],
        "next_gate": "Frozen forward shadow and empirical open-fill evidence; no PAPER/LIVE wiring.",
    }
    write_json(state_path, state_dict)

    hypothesis_path = STUDY_DIR_PATH / "hypothesis_registry.json"
    hypothesis_dict = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    hypothesis_dict["updated_at"] = finished_at_str
    for hypothesis_obj in hypothesis_dict["hypotheses"]:
        hypothesis_obj["status"] = "completed"
        hypothesis_id_str = str(hypothesis_obj["hypothesis_id"])
        if hypothesis_id_str == "H0":
            hypothesis_obj["experiment_ids"] = ["E0001", "E0002"]
            hypothesis_obj["disposition"] = (
                "supported" if verdict_dict["all_passed"] else "not_promoted"
            )
            hypothesis_obj["evidence_summary"] = verdict_dict["verdict"]
        elif hypothesis_id_str == "H1":
            hypothesis_obj["experiment_ids"] = ["E0001"]
            hypothesis_obj["disposition"] = "mixed_evidence"
            hypothesis_obj["evidence_summary"] = (
                "The frozen neighborhood remained broadly positive, but validation "
                "Sharpes ranged from 0.352 to 0.752 and were not uniformly stable."
            )
        else:
            hypothesis_obj["experiment_ids"] = ["E0001", "E0002"]
            hypothesis_obj["disposition"] = "partially_supported"
            hypothesis_obj["evidence_summary"] = (
                "Reduced beta and 82.8% exposure explain material risk reduction; "
                "the exact rule still exceeded the best static Sharpe by only about "
                "0.04 in each source-unseen period."
            )
        hypothesis_obj["data_periods_seen"] = [
            "source_discovery",
            "validation",
            "confirmation",
        ]
    write_json(hypothesis_path, hypothesis_dict)


def write_manifest() -> None:
    internal_path_list = [
        file_path
        for file_path in STUDY_DIR_PATH.rglob("*")
        if file_path.is_file() and file_path.name != "run_manifest.json"
    ]
    external_path_list = [
        NOTEBOOK_PATH,
        ALPHA_ROOT_PATH / "strategies" / "momentum" / "strategy_mo_spy_adaptive_momentum_regime.py",
        ALPHA_ROOT_PATH / "scripts" / "research" / "run_spy_adaptive_momentum_regime_study.py",
        ALPHA_ROOT_PATH / "tests" / "test_strategy_mo_spy_adaptive_momentum_regime.py",
        ALPHA_ROOT_PATH / "tests" / "test_spy_adaptive_momentum_regime_study.py",
        SOURCE_PART_1_PATH,
        SOURCE_PART_2_PATH,
    ]
    internal_record_list = []
    for file_path in sorted(set(internal_path_list), key=lambda path_obj: str(path_obj).lower()):
        internal_record_list.append(
            {
                "path": file_path.relative_to(STUDY_DIR_PATH).as_posix(),
                "sha256": sha256_file_str(file_path),
                "bytes": file_path.stat().st_size,
                "role": "material_artifact",
            }
        )
    external_record_list = []
    for file_path in sorted(set(external_path_list), key=lambda path_obj: str(path_obj).lower()):
        external_record_list.append(
            {
                "path": str(file_path),
                "sha256": sha256_file_str(file_path),
                "bytes": file_path.stat().st_size,
                "role": (
                    "external_source"
                    if file_path in {SOURCE_PART_1_PATH, SOURCE_PART_2_PATH}
                    else "external_deliverable"
                ),
            }
        )
    manifest_dict = {
        "schema_version": "quant-research-manifest-v1",
        "study_id": STUDY_ID_STR,
        "created_at": now_iso_str(),
        "data_snapshot": json.loads(DATA_METADATA_PATH.read_text(encoding="utf-8")),
        "files": internal_record_list,
        "external_deliverables_and_sources": external_record_list,
        "excluded": ["run_stdout.log and temporary render/OCR files"],
    }
    write_json(STUDY_DIR_PATH / "run_manifest.json", manifest_dict)


def run_study() -> dict[str, object]:
    if not SPEC_PATH.exists():
        raise FileNotFoundError("Freeze and validate research_spec_frozen.json first.")
    if not DATA_METADATA_PATH.exists():
        raise FileNotFoundError("Prepare the immutable data cache first.")
    metadata_dict = json.loads(DATA_METADATA_PATH.read_text(encoding="utf-8"))
    validate_source_files_and_hashes(dict(metadata_dict["source_sha256"]))
    cache_df = load_cache_df()
    target_weight_dict = build_target_weight_dict(cache_df)
    timing_return_df = build_timing_return_df(cache_df)
    source_signal_df = compute_variant_signal_df(
        cache_df["spy_total_return_close"],
        SOURCE_VARIANT_OBJ,
    )
    baseline_metrics_df = build_baseline_metrics_df(
        cache_df,
        target_weight_dict,
        timing_return_df,
    )
    variant_metrics_df = build_variant_metrics_df(
        cache_df,
        target_weight_dict,
        timing_return_df,
    )
    timing_attribution_df = build_timing_attribution_df(
        cache_df,
        target_weight_dict["source_exact"],
        timing_return_df,
    )
    state_evidence_df = build_state_evidence_df(
        cache_df,
        source_signal_df,
        timing_return_df,
    )
    capacity_df = build_capacity_df(
        cache_df,
        target_weight_dict["source_exact"],
    )
    verdict_dict = evaluate_verdict_dict(baseline_metrics_df)
    replication_dict = replication_outcome_dict(baseline_metrics_df)
    table_dict = {
        "baseline_metrics.csv": baseline_metrics_df,
        "variant_metrics.csv": variant_metrics_df,
        "timing_attribution.csv": timing_attribution_df,
        "state_evidence.csv": state_evidence_df,
        "capacity.csv": capacity_df,
        "promotion_gates.csv": verdict_dict["gate_df"],
        "source_replication.csv": replication_dict["comparison_df"],
    }
    save_tables(table_dict)
    save_charts(
        cache_df,
        target_weight_dict,
        timing_return_df,
        variant_metrics_df,
        source_signal_df,
    )
    write_reports(
        baseline_metrics_df,
        variant_metrics_df,
        timing_attribution_df,
        state_evidence_df,
        capacity_df,
        verdict_dict,
        replication_dict,
    )
    write_knowledge_record(
        baseline_metrics_df,
        capacity_df,
        verdict_dict,
        replication_dict,
    )
    write_executed_notebook()
    update_final_state(verdict_dict, replication_dict)
    write_manifest()
    return {
        "study_id": STUDY_ID_STR,
        "replication_outcome": replication_dict["outcome"],
        "research_status": verdict_dict["research_status"],
        "verdict": verdict_dict["verdict"],
        "failed_gate_count": int((~verdict_dict["gate_df"]["passed"]).sum()),
        "report": str(STUDY_DIR_PATH / "REPORT.md"),
    }


def parse_args(argument_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    action_group_obj = parser_obj.add_mutually_exclusive_group(required=True)
    action_group_obj.add_argument("--prepare-data", action="store_true")
    action_group_obj.add_argument("--write-contract", action="store_true")
    action_group_obj.add_argument("--amend-capacity-contract", action="store_true")
    action_group_obj.add_argument("--amend-timing-contract", action="store_true")
    action_group_obj.add_argument("--run", action="store_true")
    parser_obj.add_argument(
        "--source-part-1-path",
        type=Path,
        default=SOURCE_PART_1_PATH,
        help="Path to the first source PDF (defaults to the study sources directory).",
    )
    parser_obj.add_argument(
        "--source-part-2-path",
        type=Path,
        default=SOURCE_PART_2_PATH,
        help="Path to the second source PDF (defaults to the study sources directory).",
    )
    parser_obj.add_argument(
        "--pasted-note-path",
        type=Path,
        default=PASTED_NOTE_PATH,
        help="Path to the user commentary source text.",
    )
    return parser_obj.parse_args(argument_list)


def configure_source_paths(args_obj: argparse.Namespace) -> None:
    """Apply explicit, portable source paths before writing lineage artifacts."""

    global SOURCE_PART_1_PATH, SOURCE_PART_2_PATH, PASTED_NOTE_PATH
    SOURCE_PART_1_PATH = Path(args_obj.source_part_1_path).expanduser().resolve()
    SOURCE_PART_2_PATH = Path(args_obj.source_part_2_path).expanduser().resolve()
    PASTED_NOTE_PATH = Path(args_obj.pasted_note_path).expanduser().resolve()


def main() -> None:
    args_obj = parse_args()
    configure_source_paths(args_obj)
    if args_obj.prepare_data:
        result_obj = prepare_data_cache()
    elif args_obj.write_contract:
        write_frozen_contract()
        result_obj = {"study_id": STUDY_ID_STR, "contract": str(SPEC_PATH)}
    elif args_obj.amend_capacity_contract:
        amend_capacity_data_contract()
        result_obj = {"study_id": STUDY_ID_STR, "amended_contract": str(SPEC_PATH)}
    elif args_obj.amend_timing_contract:
        amend_timing_attribution_contract()
        result_obj = {"study_id": STUDY_ID_STR, "amended_contract": str(SPEC_PATH)}
    else:
        result_obj = run_study()
    print(json.dumps(result_obj, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
